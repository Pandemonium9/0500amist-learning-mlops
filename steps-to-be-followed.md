################################################################################################# 
# Tools to be Installed
#################################################################################################

# 1. Install Visual Studio Code (Editor)
- Download and install Visual Studio Code from [https://code.visualstudio.com/](https://code.visualstudio.com/).

# 2. Install Git (Version Control System)
- Download and install Git from [https://git-scm.com/](https://git-scm.com/).

# 3. Set up GitHub (For Git Repositories)
- Create an account or log in to [GitHub](https://github.com/).

# 4. Install Java JDK
- Download Java JDK 17 from [Oracle's website](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html).
- Set environment variables:
   - `JAVA_HOME`: Path to the Java installation directory (e.g., `C:\Program Files\Java\jdk-17\`).
   - Add `%JAVA_HOME%\bin` to the `PATH` variable.

# 5. Install Python
- Download and install Python from [https://www.python.org/](https://www.python.org/).
- Add Python to the `PATH` (e.g., `C:\python`).

# 6. Install Jupyter Lab
- Open a terminal and run:
   ```bash
   pip install jupyterlab
   pip install notebook
   ```

# 7. Install PySpark
- Open a terminal and run:
   ```bash
   pip install pyspark
   ```

# 8. Install Numpy and Pandas
- Open a terminal and run:
   ```bash
   pip install numpy
   pip install pandas
   ```

# 9. Install Matplotlib
- Open a terminal and run:
   ```bash
   pip install matplotlib
   ```

# 10. Install Scikit-Learn
- Open a terminal and run:
   ```bash
   pip install scikit-learn
   ```

# 11. Install Flask
- Open a terminal and run:
   ```bash
   pip install flask
   ```

# To Test:
- Use [Postman](https://www.postman.com/) or `curl` to test your setup:
   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"rooms": 2, "area": 5000}'
   ```