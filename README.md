# Data Science Project



## Installation

### Linux

Clone this repository and `cd` into it

```bash
$ git clone https://github.com/charlyalizadeh/ESILV_DataScienceProject
$ cd ESILV_DataScienceProject
```

Create a virtual environment and install the `requirements.txt`.

```bash
$ python3 -m venv .venv 
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Init the database and run the flask API.

```bash
cd flaskapp
source ./scripts/init_run.sh
```

### Windows

To clone a Github repository from windows you can use [Git for Windows](https://gitforwindows.org/)

```
> git clone https://github.com/charlyalizadeh/ESILV_DataScienceProject
> cd ESILV_DataScienceProject
```

Create a virtual environment and install the `requirements.txt`.

```
> python3 -m venv .venv 
> tutorial-env\Scripts\activate.bat
> pip install -r requirements.txt
```

Init the database and run the flask API. (In Powershell)

```
cd flaskapp
> $env:FLASK_APP = "flaskr"
> $env:FLASK_ENV = "development"
> flask init-db
> flask run
```


## TODO

### Data visualization

* [X] Spectrum visualization  
* [X] Bar plot of class  
* [X] PCA and plot  
* [X] Plot on a line pixel value depending of pixel position and spectrum

### Model

* [X] Logistic regression
* [X] Decision Tree
* [X] Random forest
* [X] Random forest bagging

### Flask API

* [X] Compare multiple models
