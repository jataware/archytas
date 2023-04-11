# clear dist folder
rm -rf dist

# build and publish package to pypi
python -m build

# upload with twine, using PYPI_USERNAME and PYPI_PASSWORD environment variables
python -m twine upload  dist/* --username $PYPI_USERNAME --password $PYPI_PASSWORD