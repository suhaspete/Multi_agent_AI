## Dynamic ML Model Selector with Automated Visual Insights 🚀📊

Imagine having a personal assistant that takes your dataset, cleans it, finds the best machine learning model, and gives you beautiful, ready-to-use visualizations of what matters most — all without lifting a finger. That’s exactly what this project is built to do.

### 🔧 What This Project Does

This tool simplifies and automates key stages of machine learning:

* **Smart Model Picker** 🧠: Whether it's classification or regression, the tool uses LazyPredict to test multiple models and chooses the best fit for your data.
* **Data Cleanup** 🧹: It handles missing values, duplicate entries, and scales your features so you can focus on the insights.
* **Feature Importance** 🔍: After selecting the top-performing model, it reveals which features have the biggest impact.
* **Interactive Graphs** 📊: Automatically generates boxplots, violin plots, and scatter plots for top features.
* **Auto-Save Visuals** 🖼️: All charts are stored in a dedicated folder for easy review and reporting.

### 💻 Tech Stack

* **Python 3.7+** 🐍
* **LazyPredict** – For quick and intelligent model selection
* **Pandas** – Data wrangling
* **Scikit-learn** – For preprocessing and evaluation
* **Matplotlib & Seaborn** – To create sleek and informative visuals

---

### 🧰 Getting Started

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2. Install Dependencies

Make sure Python 3.7 or higher is installed. Then:

```bash
pip install -r requirements.txt
```

#### 3. Launch the Script

* Open in Google Colab or run it locally.
* Drop your dataset in the working folder.
* Modify this line in the script with your dataset name:

```python
df = pd.read_csv("your_dataset.csv")
```

* Run the cells step-by-step:

  * Data cleaning
  * Model selection
  * Visualization generation

#### 4. Review the Output

* See the comparison of models right in the notebook output.
* Go to the `feature_visualizations` folder to view the saved graphs.

---

### 📂 Project Structure

```
📁 your-repo-name/
├── 📂 feature_visualizations/    # Saved plots for key features
├── main_script.py                # Main logic script
├── requirements.txt              # Required Python libraries
└── README.md                     # Project overview and instructions
```

---

### 🤝 Want to Contribute?

Pull requests and feedback are more than welcome. If you have ideas to improve the visuals, make the model picker smarter, or add new features, jump in!

---

### ▶️ Try It Live

Explore a working version on Colab:
[Launch Demo Notebook](https://colab.research.google.com/drive/1Ej8Gd9SiL4Ka-2HUdngiS-UO55oq-Er4?usp=sharing)

---

### 🎥 Visual Walkthrough

Here’s a preview of how it works:
[GitHub Demo Screenshot](https://github.com/user-attachments/assets/3b4927fe-3943-485b-9c0f-7fb90331c1de)

---

Whether you're a data scientist looking to speed up your workflow or a student trying to make sense of a dataset, this tool helps you jump from raw data to valuable insights — fast and fuss-free.
