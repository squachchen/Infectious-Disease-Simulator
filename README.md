# Infectious Disease Simulator 2.0
In this simulator, we use the same situation from the first simulator. 128 people that move at random in a square room. I had originally tried to implement a SEIR model. See [original model](https://pchenlab.wordpress.com/2020/05/25/simple-covid-19-infectious-disease-simulator/). SEIR was specific to COVID 19.
When I did the update to 1.0 model, I simplified every encounter to an Infection Rate probability (P) for performance reasons. See [blog](https://pchenlab.wordpress.com/2021/02/08/simple-infectious-disease-simulator-revisited/) 

This Infectious Disease Simulator is based on the Wells-Riley formula for airborne disease.  See [article:](https://publichealth.jhu.edu/2020/the-experiment-that-proved-airborne-disease-transmission)
The probability (P) of an individual becoming infected is expressed as
<img width="188" height="65" alt="wells_riley" src="https://github.com/user-attachments/assets/ff64b73d-7193-4a50-8206-13eb705f2442" />
Where:

P : Probability of infection.

I : Number of source cases (infected individuals).

q: Quantum emission rate (the "strength" of the infectiousness, measured in "quanta" per hour).

p: Pulmonary ventilation rate of the exposed person.

t: Total exposure duration (mins).

Q: Room ventilation rate with clean air

We end up fixing all the parameters so only the q -- emission rate and Q -- the ventilation are adjustible. 
As t (time) progresses, the probability will increase. 
As more individuas get infected, the probability increases. 
