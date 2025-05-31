\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile
from glob import glob
import numpy as np
import random
import os # Added for path manipulation

# Configuration
SEQ_LEN = 96
EMBED_SIZE = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 2 # Reduced from 3
DROPOUT = 0.25 # Adjusted
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
NUM_EPOCHS = 20 # Increased for scheduler
VOCAB_SIZE = None # To be set after tokenizing
CONDITION_DIM = 10 # Will be updated by len(condition_map)
OUTPUT_MIDI_PATH = "generated_piano_composition.mid" # Changed output path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. MIDI Tokenization ---
# Using REMI tokenizer configuration
TOKENIZER_CONFIG = TokenizerConfig(
    num_velocities=16, 
    use_chords=True, 
    use_tempos=True, 
    use_programs=False, 
    use_time_signatures=True,
    beat_res={(0, 4): 8, (4, 12): 4}
)
tokenizer = REMI(TOKENIZER_CONFIG)

# --- 2. Dataset Preparation ---
class MidiDataset(Dataset):
    def __init__(self, midi_files, tokenizer, seq_len, condition_map):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # All files will belong to the single "general_piano" condition
        self.general_piano_condition_label = "general_piano"
        if self.general_piano_condition_label not in condition_map:
            raise ValueError(f"'{self.general_piano_condition_label}' not found in condition_map. It should be the only key.")
        self.condition_idx = condition_map[self.general_piano_condition_label]
        
        self.all_tokens = []
        self.conditions = []

        for midi_file_path in midi_files:
            try:
                tokenized_midi_tracks = self.tokenizer(midi_file_path) # Returns a list of TokenSequence for tracks
                if not tokenized_midi_tracks:
                    print(f"Warning: Tokenizer returned empty for {midi_file_path}. Skipping.")
                    continue
                
                # Concatenate tokens from all tracks
                current_file_tokens = []
                for track_tokens in tokenized_midi_tracks:
                    current_file_tokens.extend(track_tokens.ids)
                
                if not current_file_tokens:
                    print(f"Warning: No tokens found after concatenating tracks for {midi_file_path}. Skipping.")
                    continue
                
            except Exception as e:
                print(f"Error tokenizing {midi_file_path}: {e}. Skipping.")
                continue
            
            if len(current_file_tokens) > self.seq_len:
                for i in range(0, len(current_file_tokens) - self.seq_len, self.seq_len // 2): # Overlapping sequences
                    self.all_tokens.append(current_file_tokens[i : i + self.seq_len + 1])
                    self.conditions.append(self.condition_idx) # Assign the single condition index
        
        global VOCAB_SIZE
        if not self.tokenizer.is_trained: # <-- Corrected: removed ()
            print("Warning: Tokenizer vocabulary might not be built. Ensure learn_bpe() has been called on training data.")
        VOCAB_SIZE = len(self.tokenizer)


    def __len__(self):
        return len(self.all_tokens)

    def __getitem__(self, idx):
        tokens = self.all_tokens[idx]
        condition = self.conditions[idx]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        c = torch.tensor(condition, dtype=torch.long) # Condition as a tensor
        return x, y, c

# --- 3. Model Definition ---
class ConditionalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, condition_dim):
        super(ConditionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Embedding for the condition
        self.condition_embedding = nn.Embedding(condition_dim, embed_size) 
        
        # LSTM input size will be embed_size (for token) + embed_size (for condition)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, h=None):
        # x shape: (batch_size, seq_len)
        # c shape: (batch_size) - condition index for each sequence in the batch
        
        embedded_x = self.embedding(x) # (batch_size, seq_len, embed_size)
        embedded_c = self.condition_embedding(c) # (batch_size, embed_size)
        
        # Expand condition embedding to match sequence length
        # embedded_c shape: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        embedded_c_expanded = embedded_c.unsqueeze(1).repeat(1, x.size(1), 1) # (batch_size, seq_len, embed_size)
        
        # Concatenate token embedding and condition embedding
        combined_input = torch.cat((embedded_x, embedded_c_expanded), dim=2) # (batch_size, seq_len, embed_size * 2)
        
        out, h = self.lstm(combined_input, h)
        out = self.dropout(out)
        out = self.fc(out) # (batch_size, seq_len, vocab_size)
        return out, h

# --- 4. Training Loop (Placeholder) ---
def train_model(model, dataloader, num_epochs, learning_rate, device):
    if VOCAB_SIZE is None:
        print("Error: VOCAB_SIZE not set. Tokenize data first.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Added weight_decay
    # Add a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10) # Removed verbose=True
    
    # Define a stop file. If this file is created, training will stop.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stop_file_path = os.path.join(script_dir, "STOP_TRAINING.flag")

    model.to(device)
    model.train()

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print(f"To manually stop training, create an empty file named 'STOP_TRAINING.flag' in the directory: {script_dir}")

    training_stopped_manually = False
    for epoch in range(num_epochs):
        # Check for stop file at the beginning of each epoch
        if os.path.exists(stop_file_path):
            print(f"Stop file '{stop_file_path}' detected. Stopping training.")
            try:
                os.remove(stop_file_path) # Remove the file so it doesn't affect next run
            except OSError as e:
                print(f"Warning: Could not remove stop file '{stop_file_path}': {e}")
            training_stopped_manually = True
            break # Exit the epoch loop

        total_loss = 0
        for batch_idx, (x, y, c) in enumerate(dataloader):
            x, y, c = x.to(device), y.to(device), c.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x, c) # Pass condition 'c' to the model
            
            # Reshape output and target for CrossEntropyLoss
            # Output: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # Y: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = criterion(output.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Added gradient clipping
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0: # Print progress
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if training_stopped_manually: # If already stopped from within batch loop (if that was added) or epoch start
            break

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss) # Step the scheduler

    # Final cleanup for the stop file in case it was created right at the end
    if not training_stopped_manually and os.path.exists(stop_file_path):
        print(f"Stop file '{stop_file_path}' detected after training loop. Cleaning up.")
        try:
            os.remove(stop_file_path)
        except OSError as e:
            print(f"Warning: Could not remove stop file '{stop_file_path}': {e}")
            
    if training_stopped_manually:
        print("Training was manually stopped.")
    else:
        print("Training finished.")

# --- 5. Music Generation (Placeholder) ---
def generate_music(model, tokenizer, start_tokens, condition_idx, max_len, device, temperature=1.0, top_p=0.0): # Added top_p, default 0.0 to be inactive unless specified
    if VOCAB_SIZE is None:
        print("Error: VOCAB_SIZE not set. Train model or load vocab first.")
        return None

    model.to(device)
    model.eval()
    
    generated_tokens = list(start_tokens) # Make a mutable copy
    current_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long).to(device)
    condition_tensor = torch.tensor([condition_idx], dtype=torch.long).to(device) # Condition for generation

    hidden = None # Or initialize if your model needs specific hidden state init

    print(f"Generating music with condition index: {condition_idx}, temp: {temperature}, top_p: {top_p}...")
    with torch.no_grad():
        for _ in range(max_len - len(start_tokens)):
            # If current_tokens_tensor becomes too long, only use the last part relevant for LSTM state
            if current_tokens_tensor.size(1) > SEQ_LEN: # Or a different relevant context window
                 input_seq = current_tokens_tensor[:, -SEQ_LEN:]
            else:
                 input_seq = current_tokens_tensor

            output, hidden = model(input_seq, condition_tensor, hidden)
            
            # Get the logits for the last token and apply temperature
            logits = output[:, -1, :] / temperature

            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask: keep at least one token if top_p is small,
                # but ensure that the first token (most probable) is always kept.
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # Clone logits to avoid modifying the original tensor if it's used elsewhere
                logits_clone = logits.clone() 
                logits_clone[0, indices_to_remove] = -float('Inf') # Apply to the batch dimension (0)
                probabilities = torch.softmax(logits_clone, dim=-1)
            else:
                # Fallback to standard softmax if top_p is not active (0.0 or >=1.0)
                probabilities = torch.softmax(logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probabilities, 1).item()
            
            generated_tokens.append(next_token)
            
            # Update current_tokens_tensor for the next iteration
            current_tokens_tensor = torch.cat((current_tokens_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)), dim=1)

            # Check if the tokenizer has the id_to_token mapping and if next_token is in it
            if hasattr(tokenizer, 'id_to_token') and next_token in tokenizer.id_to_token:
                if tokenizer.id_to_token[next_token] == "EOS_None": # End Of Sequence
                    print("End of sequence token generated.")
                    break
            elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, list) and next_token < len(tokenizer.vocab):
                # Fallback for some tokenizer types that might use a simple list vocab
                if tokenizer.vocab[next_token] == "EOS_None":
                    print("End of sequence token generated (via list vocab).")
                    break
            else:
                # This case should ideally not be hit if tokenizer is well-formed and token is valid
                # Consider adding a warning or different handling if EOS can't be checked
                pass
                
    return generated_tokens

# --- 6. Main Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Define script directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading (for training) ---
    # Simplified condition map for a single "general_piano" style
    condition_map = {
        "general_piano": 0,
    }
    CONDITION_DIM = len(condition_map) # This will be 1

    # User: Point this to your dataset of PIANO MIDI files.
    # All files in this folder will be treated as "general_piano".
    base_data_path = os.path.join(script_dir, "piano_data")
    # Load .mid files recursively from the piano_data folder and its subdirectories
    training_midi_files = []
    for root, dirs, files in os.walk(base_data_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                training_midi_files.append(os.path.join(root, file))

    if not training_midi_files:
        print(f"No MIDI files found recursively in {base_data_path}. Please check the path and ensure MIDI files are present.")
        # It's okay to proceed if we only intend to generate with a pre-trained model and a motif,
        # but training will not be possible.
        # exit() # Decide if exiting is appropriate if no training files are found.

    print(f"Found {len(training_midi_files)} MIDI files for training in {base_data_path}.")

    # --- Tokenize all MIDI files once to build vocabulary (Done during/before training) ---
    # It's crucial that the tokenizer is trained on the data it will process.
    # If VOCAB_SIZE is 0 or tokenizer is not trained, learn_bpe should be called.
    if not tokenizer.is_trained and training_midi_files: # <-- Corrected: removed ()
        print("Tokenizer vocabulary seems empty or not trained, learning BPE from training files...")
        print("This might take a while...")
        # You can adjust vocab_size as needed.
        tokenizer.learn_bpe(vocab_size=1000, files_paths=training_midi_files) 
        VOCAB_SIZE = len(tokenizer)
        # IMPORTANT: In a real workflow, save your tokenizer after learn_bpe:
        # tokenizer.save_params(os.path.join(base_data_path, "tokenizer_config.json"))
        # And load it next time:
        # tokenizer = REMI.from_file(os.path.join(base_data_path, "tokenizer_config.json"))
    elif tokenizer.is_trained: # <-- Corrected: removed ()
        VOCAB_SIZE = len(tokenizer)

    else: # No training files and tokenizer not trained
        print("Warning: Tokenizer is not trained and no training files found. VOCAB_SIZE will be 0.")
        VOCAB_SIZE = 0


    if VOCAB_SIZE == 0 and not training_midi_files:
         print("Error: Tokenizer vocabulary is empty and no training files to learn from. Cannot proceed with model initialization.")
         exit()
    elif VOCAB_SIZE == 0 and training_midi_files:
        print("Error: Tokenizer vocabulary is still empty even after attempting to learn BPE. Check training files and tokenizer.")
        exit()
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # --- Create Dataset and DataLoader (for training the base model) ---
    dataset = MidiDataset(training_midi_files, tokenizer, SEQ_LEN, condition_map)
    if len(dataset) == 0 and training_midi_files: # Only an issue if there were files but dataset is empty
        print("Training dataset is empty despite having MIDI files. Check MIDI processing, tokenization, and sequence length.")
        # exit() # Allow to proceed if only doing generation with a pre-trained model
    elif not training_midi_files:
        print("No training files, so dataset is empty. Model training will be skipped.")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) if len(dataset) > 0 else None

    # --- Initialize Model ---
    model = ConditionalLSTM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, CONDITION_DIM)
    print(model)

    # --- Train Model ---
    if dataloader and len(dataset) > 0:
        print("Starting model training...")
        train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE, device)
        # Consider saving the trained model:
        # torch.save(model.state_dict(), os.path.join(base_data_path, "trained_piano_model.pth")) # Example save path
    else:
        print("Skipping training as no data is available or dataset is empty.")
        # If you have a pre-trained model, load it here:
        # model.load_state_dict(torch.load(os.path.join(base_data_path, "trained_piano_model.pth")))
        # model.to(device) 
    # print("Skipping training for now. Ensure you have a trained model or uncomment training block.")


    # --- Motif Definition and Generation ---
    print("\\n--- Motif-Based Piano Generation ---")
    
    # USER: REPLACE THIS with the path to your actual piano motif MIDI file.
    # This file should be a short piano piece you want the model to continue.
    # It should be located in a place accessible by the script.
    # For example, you could place it in the `piano_data` folder or elsewhere.
    motif_midi_file_path = os.path.join(script_dir, "midi_export.mid") # Assumes midi_export.mid is in the same directory as the script
    # motif_midi_file_path = "c:/path/to/your/chosen/motif.mid" # Or an absolute path

    print(f"Attempting to load motif from: {motif_midi_file_path}")
    print("User: Please ensure this path points to your desired piano motif MIDI file.")

    motif_tokens = []
    if os.path.exists(motif_midi_file_path):
        try:
            motif_token_sequences = tokenizer(motif_midi_file_path) # Returns a list of TokenSequence
            if motif_token_sequences:
                # Concatenate tokens from all tracks of the motif
                for track_tokens in motif_token_sequences:
                    motif_tokens.extend(track_tokens.ids)
                
                if motif_tokens:
                    print(f"Motif loaded and tokenized into {len(motif_tokens)} tokens (from all tracks).")
                    # print(f"Motif events: {tokenizer.ids_to_events(motif_tokens)}")
                else:
                    print(f"Warning: No tokens found after concatenating tracks for motif MIDI: {motif_midi_file_path}")
            else:
                print(f"Warning: Could not tokenize motif MIDI file (tokenizer returned empty): {motif_midi_file_path}")
        except Exception as e:
            print(f"Error loading or tokenizing motif MIDI '{motif_midi_file_path}': {e}")
    else:
        print(f"Motif MIDI file not found at: {motif_midi_file_path}. Please provide a valid path.")

    if not motif_tokens:
        print("Motif is empty or could not be loaded. Using a default short sequence for demonstration.")
        # Fallback: use a very generic start if motif loading fails
        if len(dataset) > 0: # Try to get from dataset if available
             sample_x, _, _ = dataset[0]
             motif_tokens = sample_x[:10].tolist() 
             print(f"Using fallback motif from dataset sample: {tokenizer.ids_to_events(motif_tokens)}")
        elif VOCAB_SIZE > 4 : # Absolute fallback if tokenizer has some known tokens
            # Create a very basic sequence if tokenizer has some known tokens
            default_event_tokens = ["Bar_None", "Position_1/16", "Pitch_60", "Velocity_80", "Duration_1.0"]
            motif_tokens = [tokenizer.token_to_id(t) for t in default_event_tokens if tokenizer.token_to_id(t) is not None]
            if not motif_tokens or len(motif_tokens) < 2 : # If above fails, use raw indices
                 motif_tokens = [0,1,2,3,4] if VOCAB_SIZE > 4 else [] 
            print(f"Using absolute fallback motif tokens: {motif_tokens}")
        else:
            print("Cannot create a fallback motif as tokenizer vocabulary is too small or dataset is empty.")
    
    # The style for generation will be "general_piano"
    target_style_label = "general_piano"
    # Get the condition index from our simplified map
    target_style_idx = condition_map.get(target_style_label) 
    
    if target_style_idx is None:
        print(f"Error: The target style '{target_style_label}' is not in the condition_map. This should not happen with the current setup.")
        # Fallback to 0 if something went wrong, though it's unexpected.
        target_style_idx = 0 
    
    print(f"Will attempt to continue motif in style: {target_style_label} (index: {target_style_idx})")
    
    # Ensure motif_tokens has a reasonable length for generation
    if len(motif_tokens) < 2:
        print("Warning: Motif tokens length is less than 2, which might not be musically meaningful.")

    # --- Music Generation ---
    # Generate continuation with the trained model
    generated_tokens_continuation = [] # Renamed to avoid confusion with motif_tokens
    if len(motif_tokens) > 0 and VOCAB_SIZE > 0: # Check VOCAB_SIZE as well
        print("Generating music continuation for the loaded motif...")
        generated_tokens_continuation = generate_music(model, 
                                                      tokenizer, 
                                                      motif_tokens, # Pass the original motif tokens as start_tokens
                                                      target_style_idx, 
                                                      max_len=len(motif_tokens) + 500, # Generate 200 tokens *after* the motif
                                                      device=device, 
                                                      temperature=0.9, 
                                                      top_p=0.8)
    else:
        print("No valid motif tokens available or vocabulary is empty, skipping generation.")

    # Post-process and save the generated MIDI using miditok's conversion
    # The generate_music function already returns the full sequence (motif + continuation)
    # So, we use generated_tokens_continuation directly if it's not None
    if generated_tokens_continuation: 
        print(f"Generated {len(generated_tokens_continuation)} total tokens (motif + continuation).")
        try:
            # Convert token IDs to a MIDI score object using miditok
            generated_score = tokenizer.tokens_to_midi([generated_tokens_continuation])

            # Save the generated score 
            # OUTPUT_MIDI_PATH is defined globally. If it's just a filename, it saves to the current working directory.
            # If you want it in base_data_path, it should be os.path.join(base_data_path, OUTPUT_MIDI_PATH)
            # For consistency with earlier parts of the script, let's ensure it's clear.
            # The original script seemed to imply OUTPUT_MIDI_PATH was just a name, and saving happened
            # in a context where base_data_path might be relevant or just the CWD.
            # Let's use the global OUTPUT_MIDI_PATH directly as intended by its definition.
            
            save_path = os.path.join(script_dir, OUTPUT_MIDI_PATH) # Save in the script's directory
            # If OUTPUT_MIDI_PATH is not an absolute path, and you intend it to be in the script's directory or a specific one:
            # script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets directory of the script
            # save_path = os.path.join(script_dir, OUTPUT_MIDI_PATH) # Saves in the same dir as script

            generated_score.dump_midi(save_path)
            print(f"Generated motif-based piano MIDI saved to {save_path}")

        except Exception as e:
            import traceback
            print(f"Error during MIDI file creation using miditok: {e}")
            print(traceback.format_exc())
            print(f"Generated tokens were: {generated_tokens_continuation}")
    else:
        print("No tokens generated or motif generation failed, skipping MIDI file creation.")

    print("Done.")
