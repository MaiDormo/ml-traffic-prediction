# TODOS

- decide what type of traffic to pass between hosts (4 hosts)
  - could be video streaming, simple http load or some other apps
- decide maybe to genearate some traffic following a predefined signal
- understand how to format wireshark
- understand how to use gluttonts
- add maybe prophet and some other (in house solutions, or other)
  - develop a interface in order to better handle the various data formats.
- create a mini pipeline for collection and analysis

Mininet (star.py) → Ryu Controller (simple_switch.py + traffic_collector.py) 
                         ↓
                   CSV/Parquet logs
                         ↓
              Feature Engineering (preprocessor.py)
                         ↓
           ML Models (GluonTS, Prophet, LSTM)
                         ↓
              Evaluation & Plotting (analyzer.py)