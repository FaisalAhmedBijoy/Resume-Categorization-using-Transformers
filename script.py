import os
import fitz 
import shutil
import torch
import pandas as pd
import sys
from transformers import BertTokenizer, BertForSequenceClassification

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def resume_report_generation(model,tokenizer,resume_csv_path,resume_pdf_file_path,output_resume_pdf_file_path,output_categorized_resumes):
    resume_df = pd.read_csv(resume_csv_path)
    categories = resume_df['Category'].unique()
    print('categories: \n',categories)

    
    for category in categories:
        os.makedirs(output_resume_pdf_file_path+'/'+category, exist_ok=True)

    # resume_df = resume_df.head(10)
    categorize_resume_df=pd.DataFrame(columns=['filename','category'])
    for _, row in resume_df.iterrows():
        resume_id = row['ID']
        # print('Resume ID: ',resume_id)
        pdf_filename = f"{resume_id}.pdf"
        print('Filename',pdf_filename)
        pdf_path=resume_pdf_file_path+'/'+pdf_filename
        resume_text = extract_text_from_pdf(pdf_path)

        # model prediction
        inputs = tokenizer(resume_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
       
        # csv report generation
        predicted_label=categories[predicted_class_id]
        categorize_resume_df.loc[categorize_resume_df.shape[0]]=[resume_id,predicted_label]
        # print('Category: ',predicted_label)

        # copy files
        target_folder = output_resume_pdf_file_path
        source_path=pdf_path
        destination_path=target_folder+'/'+predicted_label+'/'+pdf_filename
        shutil.copyfile(source_path, destination_path)
        
    categorize_resume_df.to_csv(output_categorized_resumes,index=False)
    print('FIle copy complete and CSV file saved! ')

if __name__=='__main__':
    
    
    default_pdf_dir = 'dataset/all_resume_PDFs'
    if len(sys.argv) > 1:
        resume_pdf_file_path = sys.argv[1]
        print(f"Given path : {resume_pdf_file_path}")
    else:
        resume_pdf_file_path = default_pdf_dir
        print(f"No path provided. Using default path: {default_pdf_dir}")
   
    model_path='models/resume_classification_model'
    resume_csv_path='dataset/Resume/Resume.csv'
    resume_pdf_file_path='dataset/all_resume_PDFs'
    output_resume_pdf_file_path='logs/model_classified_resume'
    output_categorized_resumes='logs/categorized_resumes.csv'

    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer= BertTokenizer.from_pretrained(model_path)

    resume_report_generation(model,tokenizer,resume_csv_path,resume_pdf_file_path,output_resume_pdf_file_path,output_categorized_resumes)

    
