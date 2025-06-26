# 🖼️ AI Image Caption Generator

This project is a minimal image captioning model using CNN + LSTM, trained on the Flickr8k dataset.

## Features

- Preprocessing pipeline for images and captions
- Feature extraction using ResNet-50
- Custom PyTorch `Dataset` + `DataLoader`
- Minimal LSTM decoder
- Training via Jupyter notebook

## Project Structure

├── data/
├── models/
├── notebooks/
├── outputs/
├── utils/
├── main.py

## To Run

```bash
python -m utils.build_vocab
python -m utils.preprocess_captions
python -m utils.extract_features
```
