from distutils.core import setup

setup(
  name = 'pydeeptoy',
  packages = ['pydeeptoy'], # this must be the same as the name above
  version = '1.0.0.1',
  description = 'Python deep learning library for learning purposes',
  author = 'Kirill Panarin',
  author_email = 'kirill.panarin@gmail.com',
  url = 'https://github.com/stormy-ua/DeepLearningToy/tree/master/src/pydeeptoy', # use the URL to the github repo
  #download_url = 'https://github.com/peterldowns/mypackage/tarball/0.1', # I'll explain this in a second
  keywords = ['machine learning', 'deep learning', 'neural network'], # arbitrary keywords
  classifiers = [],
  install_requires=[
          'numpy', 'scikit-learn'
      ],
)