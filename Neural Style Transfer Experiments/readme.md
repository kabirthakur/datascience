Dataset 
  content images: final_images/content
  style images: final_images/styles
  
Experiment Files
	Experiment 0 : 0_OptimizersLoss.ipynb
	Experiment 1 : 1_alphabetaexpLBFGS.ipynb
  Experiment 2 : 2_alphabetaexpAdam.ipynb
  Experiment 3 : 3_layer_selection_for_style_weights.ipynb
  Experiment 4 : 4_rotation.ipynb
  Experiment 5 : 5_alphabetagamma.ipynb
  
Loss Visualisation : LossVisualization.ipynb
  Creates log loss plots for Experiment 0, 1, 2
  Necessary Loss Files
    ab_losses.json
    lbfgs_losses.json
    optimizers_losses.json
    
NST Modular Class
  NST.py
  
  
NOTE: The experiment were performed on Google Colab. The files are being referenced from a Shared Google Drive. Change paths in the importing section of the notebooks accordingly
    
    content_image1 = 'your path here/final_images/content/lion.jpg'
    .
    .
    .

    

