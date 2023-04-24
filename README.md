This project classifies a given email content into one of the four topics.

**Email Format:** 
- Json lines format: The data can be loaded into a list, where each item is a parseable JSON object.
- Data format: Each email has Subject, Body, Date (datatime stamp in which the email was recieved) 
- Example format: [ {"Subject": "",  "Body": "", "Date": _datetime_, "Label": ""} , {"Subject": "",  "Body": "", "Date": _datetime_, "Label": ""} ... ]

**Output Labels:**
- Classify the email as one of the four topics: ["sports", "world", "scitech", "business"]

**Dataset:**
- Train data: Contains 860 training samples used to train the model
- Test data (stand-alone): Contains 200 testing samples used exclusively to evaluate the model after training. Do not use this data during training.
