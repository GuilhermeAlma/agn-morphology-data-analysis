from skimage import filters, morphology, measure
from scipy import ndimage
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import csv
import os
import re

def calculate_gini(flux_values):
    if len(flux_values) == 0:
        return np.nan
    
    flux_values = np.sort(flux_values)
    n = len(flux_values)
    mean_flux = np.mean(flux_values)
    index = np.arange(1, n + 1)
    gini = (1 / (mean_flux * n * (n - 1))) * np.sum((2 * index - n - 1) * flux_values)
    return gini

def calculate_m20(flux_values, x_coords, y_coords):
    if len(flux_values) == 0:
        return np.nan
    
    x_c = np.sum(x_coords * flux_values) / np.sum(flux_values)
    y_c = np.sum(y_coords * flux_values) / np.sum(flux_values)

    r_i = (x_coords - x_c)**2 + (y_coords - y_c)**2
    m_i = flux_values * r_i

    m_tot = np.sum(m_i)
    sorted_indices = np.argsort(flux_values)[::-1]
    cumulative_sum = np.cumsum(flux_values[sorted_indices])
    f_tot = np.sum(flux_values)

    sum_m_i_selected = np.sum(m_i[sorted_indices][cumulative_sum < 0.2 * f_tot])
    if sum_m_i_selected > 0 and m_tot > 0:
        m20_index = np.log10(sum_m_i_selected / m_tot)
    else:
        m20_index = np.nan

    return m20_index

def calculate_filamentarity(largest_component):
    isophotal_area = np.sum(largest_component)
    y_coords, x_coords = np.nonzero(largest_component)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return np.nan

    x_c = np.mean(x_coords)
    y_c = np.mean(y_coords)
    
    distances = np.sqrt((x_coords - x_c)**2 + (y_coords - y_c)**2)
    major_axis_length = 2 * np.max(distances)
    
    circular_area = np.pi * (major_axis_length / 2)**2
    filamentarity = 1 - (isophotal_area / circular_area)
    return filamentarity

def run_source_extractor(image_path, config_file, param_file, conv_file):
    """Run Source Extractor on a single image and return the segmentation map data."""
    command = [
        "source-extractor",
        image_path,
        "-c", config_file,
        "-PARAMETERS_NAME", param_file,
        "-FILTER_NAME", conv_file,
        "-CHECKIMAGE_TYPE", "SEGMENTATION",
        "-CHECKIMAGE_NAME", "segmap.fits"
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    segmap_data = fits.getdata("segmap.fits")
    
    os.remove("segmap.fits")

    binary_mask_sextractor = (segmap_data > 0).astype(bool)

    return binary_mask_sextractor

def clean_mask(mask, min_size=10):
    cleaned_mask = morphology.remove_small_objects(mask, min_size=min_size)
    cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
    return cleaned_mask

def get_largest_component(mask):
    labels = measure.label(mask)
    largest_component = np.zeros(mask.shape, dtype=bool)
    if labels.max() != 0:  
        largest_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_component

def numerical_sort_key(value):
    numbers = re.findall(r'\d+', value)
    return tuple(map(int, numbers)) if numbers else (float('inf'),)

folder_path = input('Enter the path to the folder containing FITS files: ')
fits_files = sorted(glob(os.path.join(folder_path, '*.fits')), key=numerical_sort_key)

results = []
galaxy_metrics = {}

for file_path in fits_files:
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    non_finite_mask = ~np.isfinite(image_data)
    num_non_finite = np.sum(non_finite_mask)

    if 1 <= num_non_finite <= 100:
        image_data[non_finite_mask] = 0
    elif num_non_finite == np.size(image_data):
        print(f"All pixels in file {file_path} are non-finite. Skipping this image.")
        continue
    elif num_non_finite > 100:
        print(f"{file_path} has {num_non_finite} non-finite pixels. Skipping the image, checking manually is recommended.")
        continue

    thresholding_methods = {
        'Isodata': filters.threshold_isodata,
        'Li': filters.threshold_li,
        'Mean': filters.threshold_mean,
        'Minimum': filters.threshold_minimum,
        'Otsu': filters.threshold_otsu,
        'Triangle': filters.threshold_triangle,
        'Yen': filters.threshold_yen
    }

    ginis = {}
    m20s = {}
    filamentarities = {}
    num_methods = len(thresholding_methods) + 2

    fig, axs = plt.subplots(num_methods, 4, figsize=(20, num_methods * 5))
    fig.suptitle(f'Thresholding Methods Comparison for {os.path.basename(file_path)}', fontsize=16)

    for idx, (name, method) in enumerate(thresholding_methods.items()):
        try:
            threshold = method(image_data)
        except RuntimeError as e:
            print(f"Error with {name} method: {e}")
            results.append([os.path.basename(file_path), name, 'Error', 'Error', 'Error', 'N/A', 'N/A', 'N/A'])
            continue

        binary_mask = image_data > threshold

        min_size = 10
        cleaned_mask = clean_mask(binary_mask, min_size=min_size)
        largest_component = get_largest_component(cleaned_mask)

        galaxy_pixels = image_data[largest_component]

        flux_values = galaxy_pixels.flatten()
        flux_values = flux_values[~np.isnan(flux_values)]

        gini = calculate_gini(flux_values)
        y_coords, x_coords = np.nonzero(largest_component)
        m20 = calculate_m20(flux_values, x_coords, y_coords)
        filamentarity = calculate_filamentarity(largest_component)

        ginis[name] = gini
        m20s[name] = m20
        filamentarities[name] = filamentarity

        axs[idx, 0].imshow(image_data, cmap='gray')
        axs[idx, 0].set_title(f'Original Image - {name}')
        axs[idx, 0].axis('off')

        axs[idx, 1].imshow(binary_mask, cmap='gray')
        axs[idx, 1].set_title(f'Binary Mask - {name}')
        axs[idx, 1].axis('off')

        axs[idx, 2].imshow(largest_component, cmap='gray')
        axs[idx, 2].set_title(f'Cleaned Mask - {name}')
        axs[idx, 2].axis('off')

        axs[idx, 3].imshow(image_data * largest_component, cmap='gray')
        axs[idx, 3].set_title(f'Segmented Galaxy - {name}')
        axs[idx, 3].axis('off')

        results.append([os.path.basename(file_path), name, gini, m20, filamentarity, 'N/A', 'N/A', 'N/A'])

    # Multi-Otsu
    try:
        multiotsu_thresholds = filters.threshold_multiotsu(image_data)
        multiotsu_regions = np.digitize(image_data, bins=multiotsu_thresholds)
        binary_mask_multiotsu = (multiotsu_regions == 1)

        cleaned_mask_multiotsu = clean_mask(binary_mask_multiotsu, min_size=10)
        largest_component_multiotsu = get_largest_component(cleaned_mask_multiotsu)

        galaxy_pixels_multiotsu = image_data[largest_component_multiotsu]

        flux_values_multiotsu = galaxy_pixels_multiotsu.flatten()
        flux_values_multiotsu = flux_values_multiotsu[~np.isnan(flux_values_multiotsu)]

        gini_multiotsu = calculate_gini(flux_values_multiotsu)
        y_coords_multiotsu, x_coords_multiotsu = np.nonzero(largest_component_multiotsu)
        m20_multiotsu = calculate_m20(flux_values_multiotsu, x_coords_multiotsu, y_coords_multiotsu)
        filamentarity_multiotsu = calculate_filamentarity(largest_component_multiotsu)

        ginis['MultiOtsu'] = gini_multiotsu
        m20s['MultiOtsu'] = m20_multiotsu
        filamentarities['MultiOtsu'] = filamentarity_multiotsu

        results.append([os.path.basename(file_path), 'MultiOtsu', gini_multiotsu, m20_multiotsu, filamentarity_multiotsu, 'N/A', 'N/A', 'N/A'])

        axs[-2, 0].imshow(image_data, cmap='gray')
        axs[-2, 0].set_title('Original Image - MultiOtsu')
        axs[-2, 0].axis('off')

        axs[-2, 1].imshow(binary_mask_multiotsu, cmap='gray')
        axs[-2, 1].set_title('Binary Mask - MultiOtsu')
        axs[-2, 1].axis('off')

        axs[-2, 2].imshow(largest_component_multiotsu, cmap='gray')
        axs[-2, 2].set_title('Cleaned Mask - MultiOtsu')
        axs[-2, 2].axis('off')

        axs[-2, 3].imshow(image_data * largest_component_multiotsu, cmap='gray')
        axs[-2, 3].set_title('Segmented Galaxy - MultiOtsu')
        axs[-2, 3].axis('off')
    except ValueError as e:
        print(f"Error with MultiOtsu method: {e}")
        results.append([os.path.basename(file_path), 'MultiOtsu', 'Error', 'Error', 'Error', 'N/A', 'N/A', 'N/A'])
        continue
    
    # SExtractor segmentation
    try:
        config_file = '/mnt/c/Users/Work/Downloads/Files/default.sex'
        param_file = '/mnt/c/Users/Work/Downloads/Files/default.param'
        conv_file = '/mnt/c/Users/Work/Downloads/Files/default.conv'
        radius = 55

        binary_mask_sextractor = run_source_extractor(file_path, config_file, param_file, conv_file)

        cleaned_mask_sextractor = clean_mask(binary_mask_sextractor, min_size=10)
        largest_component_sextractor = get_largest_component(cleaned_mask_sextractor)

        galaxy_pixels = image_data[largest_component_sextractor]

        flux_values_sextractor = galaxy_pixels.flatten()
        flux_values_sextractor = flux_values_sextractor[~np.isnan(flux_values_sextractor)]
        
        gini_sextractor = calculate_gini(flux_values_sextractor)
        y_coords_sextractor, x_coords_sextractor = np.nonzero(largest_component_sextractor)
        m20_sextractor = calculate_m20(flux_values_sextractor, x_coords_sextractor, y_coords_sextractor)
        filamentarity_sextractor = calculate_filamentarity(largest_component_sextractor)

        ginis['SExtractor'] = gini_sextractor
        m20s['SExtractor'] = m20_sextractor
        filamentarities['SExtractor'] = filamentarity_sextractor

        results.append([os.path.basename(file_path), 'SExtractor', gini_sextractor, m20_sextractor, filamentarity_sextractor, 'N/A', 'N/A', 'N/A'])

        axs[-1, 0].imshow(image_data, cmap='gray')
        axs[-1, 0].set_title(f'Original Image - SExtractor')
        axs[-1, 0].axis('off')

        axs[-1, 1].imshow(binary_mask_sextractor, cmap='gray')
        axs[-1, 1].set_title(f'Binary Mask - SExtractor')
        axs[-1, 1].axis('off')

        axs[-1, 2].imshow(largest_component_sextractor, cmap='gray')
        axs[-1, 2].set_title(f'Cleaned Mask - SExtractor')
        axs[-1, 2].axis('off')

        axs[-1, 3].imshow(image_data * (largest_component_sextractor > 0), cmap='gray')
        axs[-1, 3].set_title(f'Segmented Galaxy - SExtractor')
        axs[-1, 3].axis('off')
    except Exception as e:
        print(f"Error with SExtractor: {e}")
        results.append([os.path.basename(file_path), 'SExtractor', 'Error', 'Error', 'Error', 'N/A', 'N/A', 'N/A'])
        continue

    galaxy_name = os.path.basename(file_path)
    if galaxy_name not in galaxy_metrics:
        galaxy_metrics[galaxy_name] = {'gini': [], 'm20': [], 'filamentarity': []}

    galaxy_metrics[galaxy_name]['gini'].extend(ginis.values())
    galaxy_metrics[galaxy_name]['m20'].extend(m20s.values())
    galaxy_metrics[galaxy_name]['filamentarity'].extend(filamentarities.values())

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join('Outputs/JWSTf770', f'{os.path.basename(file_path)}_thresholding_comparison.png'))
    plt.close()

for galaxy_name, metrics in galaxy_metrics.items():
    gini_std_dev = np.nanstd(metrics['gini'])
    m20_std_dev = np.nanstd(metrics['m20'])
    filamentarity_std_dev = np.nanstd(metrics['filamentarity'])

    for result in results:
        if result[0] == galaxy_name:
            result[-3] = gini_std_dev
            result[-2] = m20_std_dev
            result[-1] = filamentarity_std_dev

csv_file_path = os.path.join('Outputs/JWSTf770', 'gini_m20_filamentarity_coefficients.csv')
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File', 'Thresholding Method', 'Gini Coefficient', 'M20 Index', 'Filamentarity', 'Gini Std Dev', 'M20 Std Dev', 'Filamentarity Std Dev'])
    writer.writerows(results)

print(f"Results saved to {csv_file_path}")
