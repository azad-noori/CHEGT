**Reference:** This code is based on the work from [Contrastive Learning with Transformers for Meta-Path-Free Heterogeneous Graph Embedding]

📌 Overview 

This project implements a novel contrastive view learning approach for heterogeneous graph embeddings. 
 
🚀 Installation 
Prerequisites 

     Python 3.8+
     CUDA 11.0+ (optional, for GPU support)
     pip or conda
     

Install Dependencies 

pip install torch>=1.12.0
pip install torch-geometric>=2.1.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
 
  
🗂️ Dataset Preparation 
Supported Datasets 

The project supports three heterogeneous graph datasets: 
1. ACM Dataset 

     Nodes: Paper, Author, Subject
     Edges: Paper-Author, Paper-Subject
     Target Node: Paper (for classification)
     Classes: 3
     

2. DBLP Dataset 

     Nodes: Paper, Author, Conference, Term
     Edges: Paper-Author, Paper-Conference, Paper-Term
     Target Node: Paper (for classification)
     Classes: 4-5
     

3. IMDB Dataset 

     Nodes: Movie, Actor, Director
     Edges: Movie-Actor, Movie-Director
     Target Node: Movie (for classification)
     Classes: 3
 
🚀 Usage 
1. Prepare Dataset 

     Create the data directory in the project root
     Add your dataset folder (ACM, DBLP, or IMDB) to the data directory
     Ensure all required files are present as per the structure above
     

2. Configure Hyperparameters 

Edit utils/config.py to adjust: 

     Model hyperparameters
     Dataset selection
     Sampling parameters
     Early stopping settings

3. Run the Program 

python main.py


🙏 Acknowledgments 

     PyTorch Geometric for excellent graph neural network implementations
     Scikit-learn for clustering and evaluation metrics
     ACM, DBLP, and IMDB datasets providers
     All contributors who helped improve this project
     

//*****************************************************************************************************
📧 Contact and Support 
Getting Help 

If you encounter any issues while running the program or have questions about the implementation, please contact us at: 

Email: a-noori@tabrizu.ac.ir

