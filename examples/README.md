# Utility functions for Pulsed Laser Deposition

## pld_functions.py: 

  Include a digital form for growth condition recording and plume imaging management.

  Usage: 

  1. For condition recording only: 
    
    if __name__ == "__main__":
      app = QApplication(sys.argv)
      window = PLD_Form(version="parameter")
      window.show()
      app.exec_()

  2. With plume recording and management: 
  
    if __name__ == "__main__":
      app = QApplication(sys.argv)
      window = PLD_Form(version="plume")
      window.show()
      app.exec_()
      
