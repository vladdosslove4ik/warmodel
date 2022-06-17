# warmodel
Model of losses of the russian federation in the war in Ukraine based on statistics

# Model
The model consists of 12 linear regression models for each category of loses.

Based on the result for the test data, the error is estimated and a schema is built for each category

# Goal

To test if it possible to predict loses in future

# Tests
Two model tests have been created:

&emsp;•	standard (20% of randomly selected data for testing) 

&emsp;•	prediction for a certain period ahead - the last days (a designated number) 
are subtracted from the model testing dataset, on the basis 
of which the model tries to predict enemy losses day by day, 
then builds a schedule and estimates errors.

&emsp;• once the result are ready, there are 12 diagrams built and results of prediction with estimated 
percentage error is printed to console.

&emsp;• error estimation is based only known results. However, you can predict error percentage 
by estimating error to day model using current dataset

# Additional
In addition to the data, there is also an important day of war, 
because the psychological phenomena that occur with 
the time may affect.

# Results and limitations:
The overall results of the model are approximate to reality in the first 7 days, however, 
the accuracy of the model will deteriorate depending on the geometric dependence, 
which can be seen in the diagram

In theory, the model can predict the cyclicality of losses.

The linear regression model cannot predict random factors or immediate changes in the theater of war.
That defect can be eliminated only by applying the worst scenario on estimating error


# Glory to Ukraine!