# dipl
Tested on Python 3.6.7  
gym_puyopuyo env from: https://github.com/frostburn/gym_puyopuyo/  
Note: the PuyoPuyo environment posted here is slightly edited for one specific example.  
All credit goes to user [frostburn](https://github.com/frostburn)

Requires:  
`pip3 install neat-python graphviz matplotlib numpy`

Some of these (like graphviz) might require a package installation to work properly  
`sudo apt install graphviz matplotlib`  

Before installing dependancies, it is recommended to set up a virtual environment:  
`python3 -m venv /path/to/dir`  

After that activate the environment:  
`source bin/activate`

# Puyo puyo
Requires wheel for `gym` installation.  
`pip3 install wheel`  
`pip3 install -e .`

To run:  
`python3 evolve-feedforward-small.py`

# Ice slide
A sliding maze. The player is required to reach the finish point by moving in one of the 4 directions.  
Selecting a direction will move the player in that direction until he reaches an obstacle.  
Character definitions:  
`_` - empty space the player can move on  
`#` - obstacle (wall)  
`P` - player  
`X` - finish point (goal)  

