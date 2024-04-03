1.	This program is for Candidate Elimination Algorithm. It is a concept of learning algorithm in Machine Learning. It gives the result of both specific hypotheses and general hypotheses based on the input data.
2.	The input data is provided as CSV file.
3.	The programming language used is Python. Required libraries are NumPy, Pandas, Stream lit.
4.	Installation:
    •	To run the python code, Python needs to be installed.
    •	Import NumPy and pandas’ libraries. 
5.	The data set is given in the CSV format. The last column should represent the target concept, and the preceding columns should represent the attributes or features of the concepts. 
6.	Execute the Python file (app.py) and pass the path of the dataset as an argument.
7.	   "Python app.py, trainingdata.csv"
8.	The script will display the initialization of specific and general hypotheses, followed by the iteration steps of the algorithm. Finally, it will print the final specific and general hypotheses.
9.	Algorithm Overview:
    •	Initialize the specific hypothesis with the first instance in the dataset.
    •	Initialize the general hypothesis with all attributes set to "?".
    •	Iterate through each instance in the dataset:
        1.	If the target concept is "yes", update the specific and general hypotheses accordingly.
        2.	 If the target concept is "no", update the general hypothesis accordingly.
    •	Remove any redundant hypotheses from the final general hypothesis.


Streamlit Link:
https://cea123.streamlit.app/

Medium Link:
https://medium.com/@vaishnavisathiyamoorthy/candidate-elimination-algorithm-4c05b344fdac
