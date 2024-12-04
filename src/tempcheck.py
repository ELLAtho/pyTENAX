# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:27:51 2024

@author: Petr
"""


from scipy.stats import norm, gaussian_kde, weibull_min, skewnorm, beta

# Distributions to test
distributions = {
    "Normal": norm,
    "Weibull": weibull_min,
    "Skew-Normal": skewnorm,
    "Beta": beta
}


# Function to fit distributions and calculate PDFs
def fit_and_get_pdf(data, x, distributions):
    fitted_pdfs = {}
    for name, dist in distributions.items():
        try:
            # Fit the distribution to the data
            params = dist.fit(data)
            # Evaluate the PDF
            pdf = dist.pdf(x, *params)
            fitted_pdfs[name] = (pdf, params)
        except Exception as e:
            print(f"Could not fit {name}: {e}")
    return fitted_pdfs

for fname in file_list:
    statname = fname.split("\\")[-1]  # Extract file name
    
    # Load data
    data = pd.read_parquet(fname)["temp"].dropna()
    
    # KDE-based PDF
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    pdf_kde = kde(x)
    
    # Fit the selected distributions and calculate their PDFs
    fitted_pdfs = fit_and_get_pdf(data, x, distributions)
    
    # Plot KDE and all fitted PDFs
    plt.figure(figsize=(8, 6))
    plt.plot(x, pdf_kde, label="KDE (Observation)", color="blue", linewidth=2)
    
    for dist_name, (pdf, params) in fitted_pdfs.items():
        plt.plot(x, pdf, label=f"{dist_name} (Fitted)", linestyle="--")
    
    plt.title(f"PDF and Fitted Distributions for {statname}")
    plt.xlabel("Tx")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()