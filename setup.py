from setuptools import setup,find_packages
with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setup(name='truenet',
	version='1.0.2',
	description='DL method for WMH segmentation',
	author='Vaanathi Sundaresan',
	install_requires=install_requires,
    scripts=['truenet/scripts/truenet', 'truenet/scripts/prepare_truenet_data', 'truenet/scripts/make_WMmask_flair'],
	packages=find_packages(),
	include_package_data=True)
