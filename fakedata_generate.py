import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the wavelength range and more refined pH values
wavelengths = np.linspace(250, 500, 500)
pH_values = [3, 4, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.3, 7.8, 8, 8.4, 9, 9.6, 10]
# Generate synthetic absorbance data with realistic transitions
def generate_absorbance_data(pH, pKa, peak1, peak2, width1, width2):
    absorbance = (
        np.exp(-((wavelengths - 314) ** 2) / (2 * width1 ** 2)) * (1 / (1 + np.exp((pH - pKa) * 3))) * peak1 +
        np.exp(-((wavelengths - 406) ** 2) / (2 * width2 ** 2)) * (1 / (1 + np.exp(-(pH - pKa) * 3))) * peak2 +
        np.random.normal(0, 0.01, wavelengths.shape)
    )
    return np.clip(absorbance, 0, 3)

data = []
pKa_actual = 6.51
for pH in pH_values:
    absorbance = generate_absorbance_data(pH, pKa_actual, 1.5, 2.5, 15, 20)
    data.append(absorbance)

# Create a DataFrame
df = pd.DataFrame(data, index=pH_values, columns=wavelengths)
df.index.name = 'pH'
df.columns.name = 'Wavelength (nm)'

# Save the DataFrame to a CSV file
df.to_csv('synthetic_absorbance_data.csv')

# Plot the synthetic data
plt.figure(figsize=(12, 8))
for i, pH in enumerate(pH_values):
    plt.plot(wavelengths, data[i], label=f'pH {pH:.2f}')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Synthetic UV Spectrum of 4-Nitrophenol in Different pH Buffer Solutions')
plt.legend(title='pH', bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
plt.grid(True)
plt.show()