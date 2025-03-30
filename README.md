# LLM_fine_tuning_code_and_data

This repository contains the data, initial models, and code used in the paper [Cross-Linguistic Examination of Machine Translation Transfer Learning](https://arxiv.org/abs/2501.00045).


## Repository Contents

- **`translation_fine_tuning_code_sample.py`**: This is the script I used to run all my experiments. To use it, simply edit the variables at the top of the script to specify the desired parameters (e.g., dataset, model, hyperparameters) and run it. The script handles the entire fine-tuning process for the machine translation models across different languages.


## Dataset

The full dataset used in this study is available on my [Google Drive](https://drive.google.com/drive/folders/1HsSEtnXS6BySklBNf3yiSPeFVlddnGAn?usp=sharing). The dataset includes parallel corpora for the languages examined in the experiments. It’s recommended to download and organize the dataset as described in the code script.

The dataset includes:

- Text files for each language pair.

- Each language group has a short and a full dataset. The one used in the paper is the short one for each language pair. 

- The format is one sentence per line and two files one for source and one for target languages. So the translation of the source language sentence on line x in the source language file is on line x in the target language file. 


## Model Files

The `.bin` files for the models fine-tuned using HuggingFace’s Transformers are also available on my [Google Drive](https://drive.google.com/drive/folders/1xB5TldtNGFoEWf4-_ab94o4RygVgUdLh?usp=sharing). These models are ready for transfer learning in machine translation tasks, and you can fine-tune them further if needed.


## How to Use

1. **Clone the repository and install requirements**:

   If you haven’t already, clone the repository to your local machine:

   ```bash
   git clone https://github.com/your_username/LLM_fine_tuning_code_and_data.git
   cd LLM_fine_tuning_code_and_data
   pip install -r requirements.txt


2. **Download the Dataset and Models**:

   Download the full dataset and model files from Google Drive. Once downloaded, place them in the appropriate directories:

   - Put the dataset files in the `Data/` directory.

   - Put the `.bin` model files in the `Model/` directory.


3. **Edit the Script**:

   Open `translation_fine_tuning_code_sample.py`. At the top of the script, you’ll find several variables that you should adjust based on your experiment. The explanation for each variable is in a comment next to it. 

4. **Run the Script**:

   Once the variables are set, you can run the script with:

   ```bash
   python translation_fine_tuning_code_sample.py
   ```

   This will start the fine-tuning process, and the script will handle training the model on the provided dataset.

## Notes

- **GPU**: Fine-tuning the models requires significant computational power, and using a GPU is highly recommended. Ensure you have the appropriate CUDA drivers installed if you're using a GPU.

- **License**: This code and data are provided for research purposes. Please reference the original paper when using this code or dataset.

## References

- [Cross-Linguistic Examination of Machine Translation Transfer Learning](https://arxiv.org/abs/2501.00045)  

- HuggingFace Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
