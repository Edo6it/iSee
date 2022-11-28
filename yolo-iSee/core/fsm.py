# =================================

State = type("State", (object,), {})

class NoState(State):
    state = "NoState"
    
    def execute(self):
        print("No State")

class Walking(State):
    state = "Walking"
    
    def execute(self):
        print("Walking")

class Crossing(State):
    state = "Crossing"
    
    def execute(self):
        print("Crossing")

# =================================

class Transition(object):
    def __init__(self, toState):
        self.toState = toState

    def execute(self):
        print("Transitioning...")

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
        
        self.curState.execute()

# =================================

class Char(object):
    def __init__(self):
        self.FSM = FSM(self)

        self.FSM.states["NoState"] = NoState()
        self.FSM.states["Walking"] = Walking()
        self.FSM.states["Crossing"] = Crossing()

        self.FSM.transitions["toWalking"] = Transition("Walking")
        self.FSM.transitions["toCrossing"] = Transition("Crossing")
        self.FSM.transitions["toNoState"] = Transition("NoState")

        self.FSM.setState("NoState")

# =================================