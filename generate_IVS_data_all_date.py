import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import os

# --- Utility Functions ---

def build_smiles_for_date(df_day, r=0.0, q=0.0):
    """
    Build implied volatility smiles for a single quote date.
    
    Parameters:
        df_day (pd.DataFrame): Data for a single quote date.
        r (float): Risk-free rate (default is 0.0).
        q (float): Dividend yield (default is 0.0).
    
    Returns:
        smiles (dict): A dictionary where keys are time to expiration (T) and values are tuples of log-moneyness (k) and implied volatility (IV).
    """
    df_day = df_day.copy()

    # Spot price of the underlying asset (assume constant for the day)
    S = float(df_day['UNDERLYING_LAST'].iloc[0])

    # Calculate time to expiration (T) in years
    day = pd.to_datetime(df_day['QUOTE_DATE'].iloc[0])
    df_day['T'] = (pd.to_datetime(df_day['EXPIRE_DATE']) - day).dt.days / 365.0
    df_day = df_day[df_day['T'] > 0]  # Keep only options with positive time to expiration

    # Calculate mid prices for calls and puts to filter out bad data
    df_day['CALL_MID'] = (pd.to_numeric(df_day['C_BID'], errors='coerce') + 
                        pd.to_numeric(df_day['C_ASK'], errors='coerce')) / 2
    df_day['PUT_MID'] = (pd.to_numeric(df_day['P_BID'], errors='coerce') + 
                        pd.to_numeric(df_day['P_ASK'], errors='coerce')) / 2

    # Select implied volatilities for out-of-the-money (OTM) options
    C_IV = pd.to_numeric(df_day['C_IV'], errors='coerce')  # Call implied volatility
    P_IV = pd.to_numeric(df_day['P_IV'], errors='coerce')  # Put implied volatility
    K = pd.to_numeric(df_day['STRIKE'], errors='coerce')  # Strike prices

    # Use call IV for strikes >= spot price, and put IV for strikes < spot price
    iv = np.where(K >= S, C_IV, P_IV).astype(float)

    # Calculate log-moneyness (k) and assign implied volatility (IV)
    df_day['IV'] = iv
    df_day['k'] = np.log(K / S)

    # Apply basic quality filters to remove invalid data
    df_day = df_day[
        np.isfinite(df_day['IV']) &  # Ensure IV is finite
        np.isfinite(df_day['k']) &  # Ensure log-moneyness is finite
        (df_day['IV'] > 0.01) & (df_day['IV'] < 3.0) &  # Filter IV range
        (df_day['CALL_MID'] > 0) & (df_day['PUT_MID'] > 0)  # Ensure positive mid prices
    ].copy()

    # Group data by expiration date and build volatility smiles
    smiles = {}
    for exp, g in df_day.groupby('EXPIRE_DATE'):
        T = float(g['T'].iloc[0])  # Time to expiration
        k = g['k'].values  # Log-moneyness
        sig = g['IV'].values  # Implied volatility
        # Ensure there are enough strikes and sufficient coverage
        if len(sig) >= 6 and (np.max(k) - np.min(k) > 0.1):
            smiles[T] = (k, sig)
    return smiles


def svi_total_var(k, a, b, rho, m, s0):
    """
    SVI total variance formula.
    
    Parameters:
        k (float): Log-moneyness.
        a, b, rho, m, s0 (float): SVI parameters.
    
    Returns:
        float: Total implied variance.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s0 ** 2))


def fit_svi_slice(k, iv, T):
    """
    Fit SVI parameters for a single slice (one expiration date).
    
    Parameters:
        k (np.array): Log-moneyness values.
        iv (np.array): Implied volatilities.
        T (float): Time to expiration.
    
    Returns:
        dict: Fitted SVI parameters and optimization details.
    """
    # Convert implied volatility (iv) to total variance (w)
    # Total variance is the square of implied volatility scaled by time to expiration.
    w = (iv ** 2) * T

    # Initial guesses for SVI parameters
    k_med = np.median(k)  # Median log-moneyness (center of the smile)
    span = np.percentile(k, 90) - np.percentile(k, 10)  # Spread of log-moneyness (width of the smile)
    a0 = max(1e-6, np.median(w) * 0.5)  # Initial guess for 'a' (minimum variance)
    b0 = 0.2  # Initial guess for 'b' (slope of the skew)
    rho0 = 0.0  # Initial guess for 'rho' (asymmetry of the skew)
    m0 = k_med  # Initial guess for 'm' (location of the minimum variance)
    s00 = max(0.1 * span, 1e-3)  # Initial guess for 's0' (width of the smile)
    x0 = np.array([a0, b0, rho0, m0, s00])  # Combine all initial guesses into an array

    # Define the residual function for optimization
    def resid(x):
        """
        Residual function to minimize during optimization.
        
        Parameters:
            x (np.array): Current values of the SVI parameters [a, b, rho, m, s0].
        
        Returns:
            np.array: Residuals between the model's predicted total variance and the observed total variance.
        """
        a, b, rho, m, s0 = x
        # Ensure parameters are valid (e.g., b > 0, s0 > 0, |rho| < 1)
        if (b <= 0) or (s0 <= 0) or (abs(rho) >= 0.999):
            return 1e3 * np.ones_like(w)  # Return large residuals for invalid parameters
        # Calculate the predicted total variance using the SVI formula
        w_hat = svi_total_var(k, a, b, rho, m, s0)
        # Clamp predicted variance to avoid negative values
        w_hat = np.maximum(w_hat, 1e-10)
        # Return the residuals (difference between predicted and observed total variance)
        return w_hat - w

    # Optimize the SVI parameters using least squares
    res = least_squares(
        resid,  # Residual function
        x0,  # Initial guesses for the parameters
        method="trf",  # Trust Region Reflective algorithm
        loss="huber",  # Use the Huber loss function to reduce sensitivity to outliers
        f_scale=0.01,  # Scaling factor for the loss function
        max_nfev=10000  # Maximum number of function evaluations
    )

    # Extract the optimized parameters from the result
    a, b, rho, m, s0 = res.x

    # Return the fitted parameters and optimization details as a dictionary
    return {
        "a": a,  # Minimum variance
        "b": b,  # Slope of the skew
        "rho": rho,  # Asymmetry of the skew
        "m": m,  # Location of the minimum variance
        "s0": s0,  # Width of the smile
        "success": res.success,  # Whether the optimization was successful
        "cost": res.cost,  # Final cost (sum of squared residuals)
        "nfev": res.nfev  # Number of function evaluations
    }

def fit_svi_for_day(df, date_str):
    """
    Fit SVI parameters for all expirations on a given quote date.
    
    Parameters:
        df (pd.DataFrame): Option data.
        date_str (str): Quote date to process.
    
    Returns:
        pd.DataFrame: Fitted SVI parameters for all expirations.
    """
    df_day = df[df['QUOTE_DATE'] == date_str].copy()
    if df_day.empty:
        raise ValueError(f"No rows for QUOTE_DATE={date_str}")

    smiles = build_smiles_for_date(df_day)

    rows = []
    for T, (k, iv) in sorted(smiles.items(), key=lambda x: x[0]):
        if T <= 0:
            continue
        p = fit_svi_slice(k, iv, T)
        if p["success"]:
            rows.append({"date": date_str, "T": T, **{k_: p[k_] for k_ in ["a", "b", "rho", "m", "s0"]}, "cost": p["cost"]})
    return pd.DataFrame(rows).sort_values("T")


# --- Main Workflow ---
def main(input_path, output_path):
    """
    Main workflow to process option data and fit SVI parameters.
    
    Parameters:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the output CSV file.
    """
    # Load and preprocess data
    df = pd.read_csv(input_path)
    df.columns = [c.strip().strip('[]').replace(' ', '').upper() for c in df.columns]

    # Parse dates
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])

    # Cast numeric columns
    num_cols = [
        "UNDERLYING_LAST", "DTE", "STRIKE", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT",
        "C_IV", "P_IV", "C_DELTA", "P_DELTA", "C_BID", "C_ASK", "P_BID", "P_ASK", "C_LAST", "P_LAST"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Take last quote of the day per contract (near EOD)
    df = df.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "QUOTE_UNIXTIME"])
    eod = df.groupby(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], as_index=False).tail(1)

    # Basic quality filters
    eod["CALL_MID"] = (eod["C_BID"] + eod["C_ASK"]) / 2
    eod["PUT_MID"] = (eod["P_BID"] + eod["P_ASK"]) / 2
    eod = eod[
        (eod["C_IV"].between(0.01, 3.0)) &
        (eod["P_IV"].between(0.01, 3.0)) &
        (eod["CALL_MID"] > 0) & (eod["PUT_MID"] > 0) &
        (eod["DTE"].between(5, 365))
    ].copy()

    # Drop super-wide spreads
    eod["CALL_SPR"] = (eod["C_ASK"] - eod["C_BID"]) / eod["CALL_MID"]
    eod["PUT_SPR"] = (eod["P_ASK"] - eod["P_BID"]) / eod["PUT_MID"]
    eod = eod[(eod["CALL_SPR"] < 0.6) & (eod["PUT_SPR"] < 0.6)]

    # Fit SVI parameters for all quote dates
    all_params = []
    for d in sorted(eod['QUOTE_DATE'].unique()):
        try:
            params_day = fit_svi_for_day(eod, d)
            params_day['date'] = d
            all_params.append(params_day)
        except Exception as e:
            print(f"Skipping {d}: {e}")

    # Save results
    svi_df = pd.concat(all_params, ignore_index=True)
    svi_df.to_csv(output_path, index=False)
    print(f"SVI parameters saved to {output_path}")


# --- Entry Point ---
if __name__ == "__main__":
    # Hardcoded paths for input and output
    path_option = "/Users/apple/PROJECT/supervised-financial-sentiment/market_forecast/data/apple/option/aapl_2021_2023.csv"
    svi_param_path = "/Users/apple/PROJECT/supervised-financial-sentiment/market_forecast/svi_params_timeseries_aapl2_2021_2023.csv"

    # Call the main function with the hardcoded paths
    main(path_option, svi_param_path)