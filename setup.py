import setuptools


setuptools.setup(
    name="simplicial_laplacian",
    version="0.1",
    author="A. Serebrennikov",
    description="Unsupervised feature extraction from graph data",
    packages=['simplicial_laplacian'],
    install_requires=['numpy', 'networkx', 'tqdm', 'matplotlib', 'sympy', 'imblearn', 'sklearn']
)
