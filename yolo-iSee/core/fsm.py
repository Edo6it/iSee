from enum import Enum

# =================================
# STATES EUNM

class State(Enum):
    NoState = "NoState"
    Walking = "Walking"
    Crossing = "Crossing"
    CrossingNoTl = "CrossingNoTl"

# =================================

class Transition(object):
    def __init__(self, toState):
        self.toState = toState

    def execute(self):
        print(f"\n\n TRANSITIONING TO {self.toState} \n\n") 

# =================================

class FSM(object):
    def __init__(self, char):
        self.char = char 
        # Dictionary to store the states
        self.states = {}
        # Dictionary to store the transitions
        self.transitions = {}
        # Current state
        self.curState = None
        # Current transition
        self.curTrans = None 

    def setState(self, stateName):
        self.curState = self.states[stateName]

    def setTransition(self, transName):
        self.curTrans = self.transitions[transName]

    def execute(self):
        if(self.curTrans):
            self.curTrans.execute()
            self.setState(self.curTrans.toState)
            self.curTrans = None 

# =================================

class Char(object):
    def __init__(self):
        self.FSM = FSM(self)

        self.FSM.states["NoState"] = State.NoState.value
        self.FSM.states["Walking"] = State.Walking.value
        self.FSM.states["Crossing"] = State.Crossing.value
        self.FSM.states["CrossingNoTl"] = State.CrossingNoTl.value

        self.FSM.transitions["toWalking"] = Transition("Walking")
        self.FSM.transitions["toCrossing"] = Transition("Crossing")
        self.FSM.transitions["toNoState"] = Transition("NoState")
        self.FSM.transitions["toCrossingNoTl"] = Transition("CrossingNoTl")
        self.FSM.setState("NoState")

# =================================