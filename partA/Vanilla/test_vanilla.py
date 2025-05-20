import os
import torch
from tqdm import tqdm
from colorama import Fore, Style
import wandb
from wandb import init
from model_vanilla import Seq2Seq, Encoder, Decoder
from dataset import TransliterationDataset, collate_fn, load_pairs, build_vocab



def save_predictions(model, dataloader, output_file, pad_idx, idx2label, SOS_token_idx, EOS_token_idx, device, beam_width=1):
    """
    Save predictions from the model to a text file, decoding output sequences to strings.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            src_batch = batch[0]  # We only need the inputs
            src_batch = src_batch.to(device)

            # Generate predictions (returns List[List[int]])
            output_sequences = model.predict(
                src_batch,
                SOS_token_idx=SOS_token_idx,
                EOS_token_idx=EOS_token_idx,
                device=device,
                beam_width=beam_width
            )

            for seq in output_sequences:
                # Stop at EOS and skip PAD/SOS/EOS
                if EOS_token_idx in seq:
                    seq = seq[:seq.index(EOS_token_idx)]
                decoded = ''.join([idx2label[idx] for idx in seq if idx not in {pad_idx, SOS_token_idx, EOS_token_idx}])
                predictions.append(decoded)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in predictions:
            f.write(line + '\n')

    print(f"✅ Saved {len(predictions)} predictions to: {output_file}")


import torch
import random
import pandas as pd
import os
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import wandb
from tqdm import tqdm

def display_model_predictions(model, test_loader, input_vocab, output_vocab, device, 
                              start_token_id, end_token_id, beam_width=None, num_samples=10,
                              max_len=50, save_all=True, output_dir="predictions_vanilla"):
    """
    Display sample predictions from the test set and save all predictions to a file.
    
    Args:
        model: The trained model for inference
        test_loader: DataLoader containing test data
        input_vocab: Dictionary mapping characters to indices for input
        output_vocab: Dictionary mapping characters to indices for output
        device: Device to run inference on
        start_token_id: Index of the start token
        end_token_id: Index of the end token
        beam_width: Width for beam search (None for greedy)
        num_samples: Number of samples to display
        max_len: Maximum length for predictions
        save_all: Whether to save all predictions to a file
        output_dir: Directory to save predictions to
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Create reverse vocabulary maps
    input_idx2char = {idx: char for char, idx in input_vocab.items()}
    output_idx2char = {idx: char for char, idx in output_vocab.items()}
    
    # Initialize colorama for terminal colors
    init()
    
    # For storing all results
    all_results = []
    display_samples = []
    
    # Generate predictions
    with torch.no_grad():
        for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(tqdm(test_loader, desc="Generating predictions")):
            src = src.to(device)
            tgt = tgt.to(device)
            batch_size = src.size(0)
            
            for i in range(batch_size):
                # Get input sequence (remove padding)
                input_seq = src[i, :src_lens[i]].cpu().numpy()
                # Get target sequence (remove padding, SOS, and EOS)
                target_seq = tgt[i, 1:tgt_lens[i]-1].cpu().numpy()
                
                # Convert indices to characters
                input_str = ''.join([input_idx2char[idx] for idx in input_seq])
                target_str = ''.join([output_idx2char[idx] for idx in target_seq])
                
                # Generate prediction using beam search if specified
                if beam_width is not None:
                    pred_indices = model.beam_search_decode(
                        src[i:i+1], src_lens[i:i+1], 
                        start_token_id, end_token_id, 
                        beam_width=beam_width, max_len=max_len
                    )
                else:
                    # Greedy decoding
                    output = model(src[i:i+1], src_lens[i:i+1], tgt[i:i+1], teacher_forcing_ratio=0)
                    pred_indices = output[0].argmax(dim=-1).cpu().numpy()
                    # Remove SOS token if present
                    if pred_indices[0] == start_token_id:
                        pred_indices = pred_indices[1:]
                
                # Convert indices to characters, stopping at EOS token
                # pred_chars = []
                # for idx in pred_indices:
                #     if idx == end_token_id:
                #         break
                #     pred_chars.append(output_idx2char.get(idx, '?'))
                # prediction = ''.join(pred_chars)

                pred_chars = []
                for idx in pred_indices:
                    # Skip start token and end token
                    if idx == start_token_id or idx == end_token_id:
                        continue
                    pred_chars.append(output_idx2char.get(idx, '?'))
                prediction = ''.join(pred_chars)
                
                # Calculate character-level accuracy
                min_len = min(len(target_str), len(prediction))
                correct_chars = sum(1 for i in range(min_len) if target_str[i] == prediction[i])
                char_accuracy = correct_chars / len(target_str) if len(target_str) > 0 else 0
                
                # Check if prediction matches target exactly
                exact_match = prediction == target_str
                
                # Store result
                result = {
                    'input': input_str,
                    'target': target_str,
                    'prediction': prediction,
                    'char_accuracy': char_accuracy,
                    'exact_match': exact_match
                }
                all_results.append(result)
                
                # If we haven't collected enough samples yet, add this one
                if len(display_samples) < num_samples:
                    display_samples.append(result)

    # Create output directory if it doesn't exist
    if save_all and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all predictions to CSV
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "test_predictions.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Saved all predictions to {csv_path}")
    
    # Display sample predictions in a creative way
    print("\n" + "="*80)
    print(f"{Fore.CYAN}💫 TRANSLITERATION MODEL PREDICTIONS 💫{Style.RESET_ALL}")
    print("="*80)
    
    # Compute overall statistics
    exact_matches = sum(1 for r in all_results if r['exact_match'])
    avg_char_acc = sum(r['char_accuracy'] for r in all_results) / len(all_results)
    
    print(f"{Fore.YELLOW}📊 OVERALL STATS:{Style.RESET_ALL}")
    print(f"  Total Samples: {len(all_results)}")
    print(f"  Exact Match Accuracy: {exact_matches/len(all_results):.2%}")
    print(f"  Avg. Character-Level Accuracy: {avg_char_acc:.2%}")
    print("-"*80)
    
    # print(f"{Fore.GREEN}🎯 SAMPLE PREDICTIONS ({num_samples}):{Style.RESET_ALL}")
    for i, sample in enumerate(display_samples):
        print(f"\n{Fore.BLUE}Example #{i+1}:{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Input (Latin): {Style.RESET_ALL}{sample['input']}")
        print(f"  {Fore.MAGENTA}Target (Devanagari): {Style.RESET_ALL}{sample['target']}")
        print(f"  {Fore.GREEN}Prediction: {Style.RESET_ALL}{sample['prediction']}")
        
        # Highlight correct and incorrect characters
        comparison = []
        for j in range(max(len(sample['target']), len(sample['prediction']))):
            if j < len(sample['target']) and j < len(sample['prediction']):
                if sample['target'][j] == sample['prediction'][j]:
                    comparison.append(f"{Fore.GREEN}{sample['prediction'][j]}{Style.RESET_ALL}")
                else:
                    comparison.append(f"{Fore.RED}{sample['prediction'][j]}{Style.RESET_ALL}")
            elif j < len(sample['prediction']):
                comparison.append(f"{Fore.RED}{sample['prediction'][j]}{Style.RESET_ALL}")
            else:
                comparison.append(f"{Fore.RED}_{Style.RESET_ALL}")
        
        print(f"  {Fore.YELLOW}Character Match: {Style.RESET_ALL}{''.join(comparison)}")
        print(f"  {Fore.CYAN}Accuracy: {Style.RESET_ALL}{sample['char_accuracy']:.2%} " + 
              (f"{Fore.GREEN}[EXACT MATCH!]{Style.RESET_ALL}" if sample['exact_match'] else ""))
    
    # Create a visualization of the results for Wandb
    plot_prediction_results(all_results, output_dir)
    
    return all_results

def plot_prediction_results(results, output_dir=None):
    """Create visualizations of prediction results"""
    
    # Extract metrics
    char_accuracies = [r['char_accuracy'] for r in results]
    exact_matches = [r['exact_match'] for r in results]
    
    # Set up plots
    plt.figure(figsize=(15, 10))
    
    # Character accuracy distribution
    plt.subplot(2, 2, 1)
    sns.histplot(char_accuracies, bins=20, kde=True)
    plt.title('Character-Level Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    
    # Word length vs accuracy
    plt.subplot(2, 2, 2)
    word_lengths = [len(r['target']) for r in results]
    plt.scatter(word_lengths, char_accuracies, alpha=0.5)
    plt.title('Word Length vs. Accuracy')
    plt.xlabel('Target Word Length')
    plt.ylabel('Character Accuracy')
    
    # Confusion matrix for first few characters
    plt.subplot(2, 1, 2)
    # Create a beautiful grid visualization of examples
    sample_size = min(20, len(results))
    samples = random.sample(results, sample_size)
    
    # For each sample, create a row showing input, target, prediction
    for i, sample in enumerate(samples):
        plt.text(0, i, f"{sample['input']}", fontsize=10, 
                 bbox=dict(facecolor='lightblue', alpha=0.3))
        plt.text(15, i, f"{sample['target']}", fontsize=10,
                 bbox=dict(facecolor='lightgreen', alpha=0.3))
        
        # Color-code the prediction based on accuracy
        color = 'green' if sample['exact_match'] else 'orange'
        plt.text(30, i, f"{sample['prediction']} ({sample['char_accuracy']:.2%})", 
                 fontsize=10, color=color, 
                 bbox=dict(facecolor='lightyellow', alpha=0.3))
    
    plt.axis('off')
    plt.title('Sample Predictions (Input → Target → Prediction)')
    
    # Save the figure
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'))
    
    # Log to Wandb
    try:
        wandb.log({"prediction_analysis": wandb.Image(plt)})
    except:
        print("Couldn't log to Wandb - may not be initialized or connected")
    
    plt.close()
    
    # Create a more visually appealing HTML report
    create_html_report(results[:100], output_dir)  # Limit to 100 samples for HTML report

def create_html_report(results, output_dir=None):
    """Create an HTML report with prediction results"""
    if not output_dir:
        return
        
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transliteration Model Predictions</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; text-align: center; }
            .stats { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .prediction-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
            .prediction-card { 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 15px; 
                transition: transform 0.2s;
            }
            .prediction-card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .match { background-color: #d4edda; }
            .close { background-color: #fff3cd; }
            .miss { background-color: #f8d7da; }
            .input { font-weight: bold; color: #0066cc; }
            .target { font-weight: bold; color: #28a745; }
            .prediction { font-weight: bold; }
            .char-match { color: green; }
            .char-miss { color: red; }
            .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .meter { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin-top: 5px; }
            .meter-fill { height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); }
        </style>
    </head>
    <body>
        <h1>🔤 Hindi Transliteration Model Results 🔤</h1>
        
        <div class="stats">
            <h2>Overall Statistics</h2>
            <div class="stats-grid">
                <div>
                    <p>Total Samples: <strong>TOTAL_COUNT</strong></p>
                    <p>Exact Match Accuracy: <strong>EXACT_MATCH_PCT%</strong></p>
                    <div class="meter">
                        <div class="meter-fill" style="width: EXACT_MATCH_PCT%"></div>
                    </div>
                </div>
                <div>
                    <p>Average Character Accuracy: <strong>AVG_CHAR_ACC%</strong></p>
                    <div class="meter">
                        <div class="meter-fill" style="width: AVG_CHAR_ACC%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <h2>Sample Predictions</h2>
        <div class="prediction-grid">
            PREDICTION_CARDS
        </div>
    </body>
    </html>
    """
    
    # Calculate statistics
    total_count = len(results)
    exact_matches = sum(1 for r in results if r['exact_match'])
    exact_match_pct = round(exact_matches / total_count * 100, 1)
    avg_char_acc = round(sum(r['char_accuracy'] for r in results) / total_count * 100, 1)
    
    # Generate prediction cards
    prediction_cards = []
    for r in results:
        # Determine card class based on accuracy
        if r['exact_match']:
            card_class = "match"
        elif r['char_accuracy'] > 0.7:
            card_class = "close"
        else:
            card_class = "miss"
            
        # Create character-by-character comparison
        char_comparison = []
        for i in range(max(len(r['target']), len(r['prediction']))):
            if i < len(r['target']) and i < len(r['prediction']):
                if r['target'][i] == r['prediction'][i]:
                    char_comparison.append(f'<span class="char-match">{r["prediction"][i]}</span>')
                else:
                    char_comparison.append(f'<span class="char-miss">{r["prediction"][i]}</span>')
            elif i < len(r['prediction']):
                char_comparison.append(f'<span class="char-miss">{r["prediction"][i]}</span>')
            else:
                char_comparison.append(f'<span class="char-miss">_</span>')
        
        card_html = f"""
        <div class="prediction-card {card_class}">
            <p><span class="input">Input:</span> {r['input']}</p>
            <p><span class="target">Target:</span> {r['target']}</p>
            <p><span class="prediction">Prediction:</span> {r['prediction']}</p>
            <p>Character Match: {''.join(char_comparison)}</p>
            <p>Accuracy: {round(r['char_accuracy']*100, 1)}%</p>
        </div>
        """
        prediction_cards.append(card_html)
    
    # Replace placeholders
    html_content = html_content.replace("TOTAL_COUNT", str(total_count))
    html_content = html_content.replace("EXACT_MATCH_PCT", str(exact_match_pct))
    html_content = html_content.replace("AVG_CHAR_ACC", str(avg_char_acc))
    html_content = html_content.replace("PREDICTION_CARDS", "\n".join(prediction_cards))
    
    # Save the HTML file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prediction_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Saved HTML report to {os.path.join(output_dir, 'prediction_report.html')}")


import torch
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
from tqdm import tqdm

# Use the display_model_predictions function we've defined
def test_and_visualize_predictions(model_path=None, num_samples=10):
    """
    Test the model and visualize its predictions on test data.
    
    Args:
        model_path: Path to the saved model (if None, use best model from Wandb)
        beam_width: Width for beam search
        num_samples: Number of samples to display
    """
    print("Initializing Wandb...")
    try:
        run = wandb.init()
        print("Wandb initialization successful.")
        beam_width=0
        # Set up Wandb API to get the best model if needed
        if model_path is None:
            ENTITY = "da24m012-iit-madras"
            PROJECT = "dakshina-transliteration"
            SWEEP_ID = "r1zmunpz"
            api = wandb.Api()
            
            # Get all runs in the project
            all_runs = api.runs(f"{ENTITY}/{PROJECT}")
            
            # Filter runs that belong to the desired sweep
            sweep_runs = [run for run in all_runs if run.sweep and run.sweep.id == SWEEP_ID]
            best_run = max(sweep_runs, key=lambda run: run.summary.get("val_acc", 0.0))
            
            # Show details of the best run
            print(f"Best run ID: {best_run.id}")
            print(f"Name: {best_run.name}")
            print(f"Best Validation Accuracy: {best_run.summary['val_acc']}")
            print("Config:", dict(best_run.config))
            
            config = dict(best_run.config)
            best_run.file("best_model.pth").download(replace=True)
            model_path = "/kaggle/working/best_model.pth"
        else:
            config = torch.load(model_path, map_location='cpu')['config']

        beam_width=config["beam_width"]
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load data and build vocabularies
        # from data_utils import load_pairs, build_vocab, TransliterationDataset, collate_fn
        
        train_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.train.tsv")
        val_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.dev.tsv")
        test_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.test.tsv")
        
        input_vocab, output_vocab = build_vocab(train_pairs)
        pad_idx = output_vocab['<pad>']
        SOS_token_idx = output_vocab['<sos>']
        EOS_token_idx = output_vocab['<eos>']
        
        # Rebuild model architecture
        # from model import Encoder, Decoder, Seq2Seq
        
        encoder = Encoder(
            len(input_vocab),
            config['embed_size'],
            config['hidden_size'],
            config['num_layers_encoder'],
            config['cell_type'],
            config['dropout']
        )
        decoder = Decoder(
            len(output_vocab),
            config['embed_size'],
            config['hidden_size'],
            config['num_layers_decoder'],
            config['cell_type'],
            config['dropout']
        )
        model = Seq2Seq(encoder, decoder, config['cell_type']).to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded and ready.")
        
        # Create test DataLoader
        test_dataset = TransliterationDataset(test_pairs, input_vocab, output_vocab)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.get('batch_size', 64), 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Create output directory
        output_dir = "predictions_vanilla"
        os.makedirs(output_dir, exist_ok=True)
        
        # Display predictions and save results
        results = display_model_predictions(
            model=model,
            test_loader=test_loader,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            device=device,
            start_token_id=SOS_token_idx,
            end_token_id=EOS_token_idx,
            beam_width=beam_width,
            num_samples=num_samples,
            output_dir=output_dir
        )
        
        # Save model config for reference
        with open(os.path.join(output_dir, "model_config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n🎉 All predictions saved to {output_dir}/")
        print(f"✨ Sample visualizations are available in {output_dir}/prediction_analysis.png")
        print(f"📊 Interactive HTML report available at {output_dir}/prediction_report.html")
        print(f"📝 Complete prediction results in {output_dir}/test_predictions.csv")
        
        # Save a summary of results to wandb
        wandb.log({"Test Accuracy": sum(r['char_accuracy'] for r in results) / len(results)})
        
        # Create additional visualizations for analysis
        plot_prediction_results(results, output_dir)
        
        return results
        
    except Exception as e:
        print(f"Error in test_and_visualize_predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test and visualize model predictions.")
    parser.add_argument("--model_path", type=str, help="Path to the saved model.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to display.")
    args = parser.parse_args()

    results = test_and_visualize_predictions(num_samples=args.num_samples, model_path=args.model_path)
