import platform
from setuptools import setup, find_packages
import os

# Define package metadata
package_name = "CRMS2Map"
version = "1.0"
description = "A package for CRMS2Map data processing and analysis"
long_description = open("README.md").read()
url = "https://github.com/jinikeda/CRMS2Map"
author = "Jin Ikeda"
license_type = "MIT License"
install_requires = []
install_requires = [
    "gdal",
    "numpy",
    "pandas",
    "geopandas",
    "rasterio",
    "scipy",
    "matplotlib",
    "basemap",
    "rioxarray",
]

# ## Determine the platform and make platform-specific adjustments
# if platform.system() in ['Darwin', 'Linux', 'Ubuntu']:
#     # Proper installation for Mac and Linux
#     crms_script = 'src.bin.crms'
# elif platform.system() == 'Windows':
#     import shutil
#     crms_script = 'src.bin.crms_script'
#     crms_win_script = 'src.bin.crms_win'
#     shutil.copyfile(crms_script.replace('.', '/'), crms_win_script.replace('.', '/') + '.py')
#     crms_script = crms_win_script
# else:
#     raise OSError(f"Unsupported platform: {platform.system()}")

# Ensure README.md is present
if not os.path.exists("README.md"):
    long_description = description  # Fallback if README.md is missing

# Setup configuration
setup(
    name=package_name,
    version=version,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "CRMS2Map_continuous=CRMS_Continuous_Hydrographic2subsets:subsets",
            "CRMS2Map_discrete=CRMS_Discrete_Hydrographic2subsets:subsets",
            "CRMS2Map_resample=CRMS2Resample:resample",
            "CRMS2Plot=CRMS2Plot:data_analysis",
        ],
    },
    author=author,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    license=license_type,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
