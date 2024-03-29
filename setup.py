from distutils.core import setup
setup(
  name = 'Singularity',
  packages = ['Singularity'],
  version = '1.0',
  license='Apache-2.0 license',      
  description = 'Simple Machine learning library',   
  author = 'Arda Aksu',                   
  author_email = 'singularity.ai.contact@gmail.com',      
  url = 'https://github.com/aarda55/Singularity', 
  download_url = 'https://github.com/aarda55/Singularity/archive/refs/tags/v.0.1.1.tar.gz',
  keywords = ['Machine Learning', 'High level API'], 
  install_requires=[            
          'numpy',
          'pickle',
          'cv2',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'License :: OSI Approved :: Apache-2.0 license',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.10',
  ],
)