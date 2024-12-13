# ImageTagger
ImageTagger: A collection of image tagging methods.
## Install
```
git clone https://github.com/RobertLau666/ImageTagger.git

conda create -n imagetagger python=3.10
conda activate imagetagger
pip install --upgrade pip

cd ImageTagger/models/LLaVA_NeXT
pip install -e ".[train]"

cd ../..
pip install -r requirements.txt
```
## Run
```
bash app.sh
```