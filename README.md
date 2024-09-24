# Password Predictor
## Some background info:
In 2021 during covid I was an IT student and looking for some fun. I decided to set up a ssh honeypot and see what kind of passwords hackers where trying on it.
Al this data can be found in [SSH-findings](https://github.com/anakwaboe4/SSH-findings) project. I copied over the main events.csv file to this project to use as a dataset.
## Goal of this project:
This project is a very simple case of classification. The goal is to predict the if a given string would be in the list of passwords tried on my ssh honeypot.
## Technical dive:
## usage:
install the requirements:
```bash
pip install -r requirements.txt
```
### manual:
run the main.py file:
```bash
python main.py
```
### API:
run the api.py file:
```bash
python api.py
```
**Note:** The API is not yet developed! 🚧

rest of the readme is still under construction

