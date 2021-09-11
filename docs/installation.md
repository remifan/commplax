# Installation

We assume you have seen command-line operations, since the following steps are command-line based.

```{admonition} Windows
commplax is built on top of Jax, which does not support Windows at this moment, so neither does commplax. If you are Windows 10+ user, you may try [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about), which is an efficient GNU/Linux compatible environment made by Microsoft.

After you have WSL and WSL/Linux (e.g., Ubuntu) installed, lauching the linux OS should bring you a terminal prompt, and that's where we starts.
```

If you are familar with Python virtual enviroment (conda, pipenv...), you may skip to [Install Jax and Commplax](#install-jax-and-commplax).


## Python Virtual Environment (venv)

### why venv?
Like most programming language, a Python program is essentially the composition of some user scripts and their dependences (packages) installed through package manager (`pip`).

For example, one is programming a GPU pricing prediction software which relies on package (`scipy`) having linear regression function that itself depends on matrix manipulation package (`numpy`), and it is nice to install a visualizaition package(`matplotlib`) which also depends on `numpy`. The entire dependences have a directed acyclic graph (DAG) structure, meaning that multiple packages might require the same package.

It could be fine at start, but as one has more projects, a new project with package `A` needs recent version of package `C` but package `B` from another earlier project needs the older version of `C`. That's when package conflicts occur, and venv tool supports creating 'workspaces' each with isolated packages resolution.

There are many tools that support venv, we use `conda` (from miniconda) for example.

#### Install miniconda
install through official scripts
- [Installing on Linux/WSL](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [Installing on MacOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

install through package manager
- RPM, Debian based: see [Conda doc](https://docs.conda.io/projects/conda/en/latest/user-guide/install/rpm-debian.html)
- Homebrew on MacOS: `brew install --cask miniconda`

now run `conda init`

#### Create virtual conda environment
the first step is to create a venv for commplax, the name and python version could be others 

`conda create --name commplax python=3.8`

the next step is to activate/enter the newly created venv by running

`conda activate commplax`

to exit the current venv

`conda deactivate`

there are many more steps you would need in daily use of conda, see [Conda's guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) for more information.


## Install Jax and Commplax
The majority of commplax functions runs very well(or even better) with CPU only, its GPU version is based on CUDA.

```{note} For regular Window 10 users, WSL currently has no support to use GPU-CUDA, see [WSLg](https://github.com/microsoft/wslg) project
```

in your activated 'commplax' venv

### Install CPU-only version
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```
### Install CPU+GPU version
you must install jaxlib that matches your cuda driver version, for example, if your CUDA version is 11.0,
```
pip install --upgrade jax==0.2.13 jaxlib==0.1.66+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

after jaxlib+cuda is installed,
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```

### Install for Development
You can work the commplax's source codes by installing commplax on "development mode"

- install Jax same as the above
- `git clone https://github.com/remifan/commplax && cd commplax`
  `pip install -e '.'`


## Development environment
Now we need development environment to work within commplax venv, we prefer notebook-based enviroment ([JupyterLab](https://jupyter.org/index.html)) to work on commplax, all the examples we provide are in the form of notebook.

[Try JupyterLab](https://jupyter.org/try) in your web browser

If it feels good, [Install JupyterLab](https://jupyter.org/install.html) through `conda`.

```{note} Though it is convenient to install JupyterLab alongside commplax in the same venv. To minimize the risk of package confilcts, the best practice is to install JupyterLab in a standalone venv, and [add other venvs to JupyterLab](https://stackoverflow.com/a/53546634)
```

