# ScholarshipProbabilityModel
A neural network that uses a student's profile to give the probability of them earning a scholarship.

# Setup
As you look through our code, you would notice that we have imported a few packages. If you do not have them installed, use the command `pip install libraryname` in the Visual Studio Code terminal. For example, installing the scikit-learn package would look like `pip install scikit-learn`. All of the packages need to be installed in order for the program to work. 

Note: If you are using the Jupyter Notebook file, you do not need to install everything separately. Simply run each cell in order.

# Execution Instructions
We highly recommend that you run each of the model's versions. It would give you a better idea of our development process and how much more capable the model has become through the different stages. The model version is named at the top of each cell in the `scholarship_probability.ipynb` file. Once you have installed the packages, you may need to download the Jupyter Notebook extension from Visual Studio Code. Finally, you can simply run the cell/program. The model should take a few seconds to train and wil return the probability, along with the accuracy and loss graphs. 

For the final version of the model, it should prompt you to enter information. For the sake of simplicity, we only kept a few keywords as inputs, which are written in a comment near the bottom of the code. To exit the program, enter `Y` when prompted.

# Final Comments
When running the final version of the ScholarshipProbability model, you would probably notice that it often returns either a value very close to 100% or 0%. This is because of the way that the synthetic data was generated. Essentially, we used a basic point system to calculate whether a student would receive a scholarship. Because of this, our data is extremely binary. In a real-world dataset, there would not be a defined set of rules for who obtains a scholarship; there is luck/randomness involved in the process. With a real dataset, our model would be able to give a more accurate representation of the user's chances. However, due to the time constraints (as it is a hackathon) and the lack of access our team has for scholarship data, we were unable to find a usable source of data. We hope you enjoy trying out our model!
