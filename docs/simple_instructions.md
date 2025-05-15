# Step-by-Step Guide to Downloading and Installing the Mental Health LLM Evaluator from GitHub

## Introduction
This guide will help you download the Mental Health LLM Evaluator repository from GitHub and set it up on your local machine. No prior experience with GitHub is required!

## Step 1: Install Git
Before you can download the repository, you need to have Git installed on your computer.

### For Windows:
1. **Download Git**: Go to the [Git for Windows website](https://gitforwindows.org/) and click on the "Download" button.
2. **Run the Installer**: Once the download is complete, run the installer and follow the prompts. You can keep the default settings.

### For macOS:
1. **Open Terminal**: You can find Terminal in your Applications folder under Utilities.
2. **Install Git**: Type the following command and press `Enter`:
   ```bash
   git --version
   ```
   If Git is not installed, you will be prompted to install it. Follow the instructions to complete the installation.

### For Linux:
1. **Open Terminal**: You can usually find Terminal in your applications menu.
2. **Install Git**: Type the following command and press `Enter`:
   ```bash
   sudo apt install git
   ```
   Enter your password if prompted, and follow the instructions to complete the installation.

## Step 2: Download the Repository
1. **Open Your Web Browser**: Use any web browser like Chrome, Firefox, or Safari.
2. **Go to the Repository**: Navigate to the [Mental Health LLM Evaluator GitHub page](https://github.com/dccooper/mental_health_llm_eval).
3. **Find the Code Button**: On the repository page, look for a green button labeled "Code" and click on it.
4. **Copy the URL**: In the dropdown menu, you will see a URL. Click on the clipboard icon to copy it.

## Step 3: Clone the Repository
1. **Open Terminal or Command Prompt**: Depending on your operating system, open the Terminal (macOS/Linux) or Command Prompt (Windows).
2. **Navigate to Your Desired Directory**: Use the `cd` command to change to the directory where you want to download the repository. For example:
   ```bash
   cd Documents
   ```
3. **Clone the Repository**: Type the following command and paste the URL you copied earlier:
   ```bash
   git clone https://github.com/dccooper/mental_health_llm_eval.git
   ```
   Press `Enter`. This will download the repository to your local machine.

## Step 4: Install Python and Required Packages
1. **Install Python**: If you don’t have Python installed, download it from the [official Python website](https://www.python.org/downloads/). Follow the installation instructions for your operating system.
2. **Open Terminal or Command Prompt**: Navigate to the directory where you cloned the repository:
   ```bash
   cd mental_health_llm_eval
   ```
3. **Create a Virtual Environment (Optional but Recommended)**: This helps manage dependencies. Run the following command:
   ```bash
   python3 -m venv venv
   ```
4. **Activate the Virtual Environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

5. **Install Required Packages**: Run the following command to install the necessary packages:
   ```bash
   pip install -e .
   ```

## Step 5: Run the Application
1. **Start the Application**: In the same terminal window, run the following command:
   ```bash
   streamlit run app/streamlit_app.py
   ```
2. **Open Your Web Browser**: After running the command, open your web browser and go to `http://localhost:8501` to access the application.

## Conclusion
  * [ ] You have successfully downloaded and installed the Mental Health LLM Evaluator! If you have any questions or need further assistance, please refer to the documentation or contact support.

---

# Step-by-Step Guide to Using the Mental Health LLM Evaluator

## Introduction
The Mental Health LLM Evaluator is a user-friendly web application designed to help mental health professionals evaluate language model responses. This guide will walk you through the process of using the tool, from setting it up to submitting evaluations.

## Step 1: Access the Application
1. **Open Your Web Browser**: Use any web browser like Chrome, Firefox, or Safari.
2. **Enter the URL**: Type `http://localhost:8501` in the address bar and press `Enter`. This will take you to the Mental Health LLM Evaluator interface.

## Step 2: Configure the Scoring Rubric (Optional)
1. **Locate the Sidebar**: On the left side of the screen, you will see the sidebar with various options.
2. **Download the Scoring Rubric Template**: Click on the button labeled "📥 Download Rubric Template" to download a CSV file. This file contains a template for scoring criteria.
3. **Modify the Template**: Open the downloaded CSV file in a spreadsheet application (like Microsoft Excel or Google Sheets) and adjust the criteria, weights, and scales as needed.
4. **Upload Your Custom Rubric**: After modifying the template, return to the sidebar and use the "📄 Upload Custom Rubric" option to upload your modified CSV file.

## Step 3: Upload a Prompt Bank
1. **Download the Prompt Template**: Click on the button labeled "📥 Download Prompts Template" to download a CSV file with example prompts.
2. **Create Your Prompt Bank**: Use the downloaded template to create your own prompt bank in CSV or YAML format.
3. **Upload Your Prompt File**: In the sidebar, find the "📄 Upload Prompts File" section. Click on it and select your prompt bank file (either CSV or YAML) from your computer.

## Step 4: Select Prompts for Evaluation
1. **Filter by Category**: If your prompts are categorized, use the "🏷️ Filter by Category" dropdown to select a specific category or choose "All" to see all prompts.
2. **Choose a Prompt**: From the "📝 Select Prompt" dropdown, select the prompt you want to evaluate. This will load the prompt into the evaluation interface.

## Step 5: Review the Evaluation
1. **Check the Model's Response**: The application will automatically generate a response from the language model based on the selected prompt.
2. **Review Detected Red Flags**: Look for any potential safety concerns highlighted in the "🚩 Detected Issues" section.
3. **Adjust Scores**: In the "Evaluation" section, you can adjust the scores for different evaluation dimensions using the sliders provided.

## Step 6: Provide Justification and Feedback
1. **Add Justification**: In the "Review" section, provide a justification for the scores you assigned. This helps explain your evaluation.
2. **Additional Feedback**: Use the "Additional Feedback" text area to add any other observations or suggestions.

## Step 7: Submit Your Evaluation
1. **Submit the Evaluation**: Once you are satisfied with your scores and feedback, click the "✅ Submit Evaluation" button to save your evaluation.
2. **View Session Statistics**: After submission, you can view session statistics in the sidebar, which will show metrics like total evaluations and average scores.

## Step 8: Export Results (Optional)
1. **Download Results**: If you want to keep a record of your evaluations, use the "📥 Download CSV" or "📥 Download JSON" buttons to export your results in your preferred format.

## Conclusion
You have now successfully used the Mental Health LLM Evaluator! If you have any questions or need further assistance, please refer to the documentation or contact support.
