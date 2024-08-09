import platform
from setuptools import setup, find_packages


# Define package metadata
package_name = "CRMS2Map"
version = "1.0" # Begining version
description = "A package for CRMS2Map data processing and analysis"
long_description = open('README.md').read()
url = "https://github.com/jinikeda/CRMS2Map"  # Replace with your actual URL
author = "Jin Ikeda",
license_type = "MIT License"

# Determine the platform and make platform-specific adjustments
if platform.system() == 'Ubuntu' or platform.system() == 'Linux' or platform.system() == 'Darwin':
    # Proper installation for Mac and Linux
    crms_script = 'src/bin/crms_script'

elif platform.system() == 'Windows':
    import shutil
    crms_script = 'src/bin/crms_script'
    crms_win_script = 'src/bin/crms_script_win.py'
    shutil.copyfile(crms_script, crms_win_script)
    crms_script = crms_win_script

else:
    raise OSError(f"Unsupported platform: {platform.system()}")

# Setup configuration
setup(
    name=package_name,
    version=version,
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        # Add your package dependencies here
    ],
    entry_points={
        'console_scripts': [
            f'{package_name}={crms_script}:main'
        ],
    },
    author=author,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)