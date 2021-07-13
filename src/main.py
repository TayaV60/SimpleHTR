import argparse
import json
from typing import Tuple, List
import datetime
import cv2
import editdistance
from path import Path
from autocorrect import Speller

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_lossvsepoch = '../model/lossvsepoch.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_error_rates: List[float], with_spell_checker_error_rates: List[float], spell_checker_improvement_rates: List[float], spell_checker_worsening_rates: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({
            'charErrorRates': char_error_rates, 
            'wordErrorRates': word_error_rates, 
            'withSpellCheckerErrorRates': with_spell_checker_error_rates, 
            'spellCheckerImprovementRates': spell_checker_improvement_rates,
            'spellCheckerWorseningRates': spell_checker_worsening_rates
        }, f)


def write_lossvsepoch(epochs: List[int], average_losses: List[float]) -> None:
    """Saves the change of average loss over epochs, done once at the end of training."""
    with open(FilePaths.fn_lossvsepoch, 'w') as f:
        json.dump({'Epochs': epochs, 'Average Loss': average_losses}, f)


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          spell,
          early_stopping: int = 15) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_error_rates = []
    with_spell_checker_error_rates = []
    spell_checker_improvement_rates = []
    spell_checker_worsening_rates = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # keep arrays of loss and epochs (for analysis)
    epochs = []
    average_losses = []
    # stop training after this number of epochs without improvement
    while True:
        losses = [] # the losses for each epoch
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            losses.append(loss)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
        
        average_loss = sum(losses) / len(losses)
        average_losses.append(average_loss)
        epochs.append(epoch)

        # validate
        char_error_rate, word_error_rate, with_spell_checker_error_rate, spell_checker_improvement_rate, spell_checker_worsening_rate = validate(model, loader, line_mode, spell)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_error_rates.append(word_error_rate)
        with_spell_checker_error_rates.append(with_spell_checker_error_rate)
        spell_checker_improvement_rates.append(spell_checker_improvement_rate)
        spell_checker_worsening_rates.append(spell_checker_worsening_rate)
        write_summary(summary_char_error_rates, summary_word_error_rates, with_spell_checker_error_rates, spell_checker_improvement_rates, spell_checker_worsening_rates)
        # this currently overwrites for each epoch, which is wasteful but might allow us to
        # track progress if the program fails to complete
        # TODO under the while loop maybe
        write_lossvsepoch(epochs, average_losses)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()

        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def _verify(model: Model, loader: DataLoaderIAM, line_mode: bool, spell) -> Tuple[float, float]:
    """Trains or Validates NN - requires the loader to have loaded its set before calling."""
    if len(loader.samples) == 0:
        raise Exception('The number samples is 0 - has the loader been loaded?')
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_err_with_spell_check = 0
    num_char_total = 0
    num_word_err = 0
    num_word_total = 0
    # a count of errors that could potentially be autocorrected correctly
    num_spell_checker_improved_result = 0
    num_spell_checker_worsened_result = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_err += 1 if batch.gt_texts[i] != recognized[i] else 0 # also TBD
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist

            spelt = spell(recognized[i])
            dist2 = editdistance.eval(spelt, batch.gt_texts[i])
            num_char_err_with_spell_check += dist2

            # if the recognised is the same as the ground truth, but the spell checker is not
            if dist2 > 0 and batch.gt_texts[i] == recognized[i]:
                num_spell_checker_worsened_result += 1

            # if the recognised is not the same as the ground truth, but the spell checker is
            if dist > 0 and spelt == batch.gt_texts[i]:
                num_spell_checker_improved_result += 1
            
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_error_rate = num_word_err / num_word_total
    with_spell_checker_error_rate = num_char_err_with_spell_check / num_word_total
    spell_checker_improvement_rate = num_spell_checker_improved_result / num_word_total
    spell_checker_worsening_rate = num_spell_checker_worsened_result / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word error rate: {word_error_rate * 100.0}%. Spell checker improvement rate: {spell_checker_improvement_rate * 100.0}%. Spell checker worsening rate: {spell_checker_worsening_rate * 100.0}%. With spell checker error rate: {with_spell_checker_error_rate * 100.0}%.')
    return char_error_rate, word_error_rate, with_spell_checker_error_rate, spell_checker_improvement_rate, spell_checker_worsening_rate


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool, spell) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    return _verify(model, loader, line_mode, spell)


def test(model: Model, loader: DataLoaderIAM, line_mode: bool, spell) -> Tuple[float, float]:
    """Trains NN."""
    print('Test NN')
    loader.test_set()
    return _verify(model, loader, line_mode, spell)


def infer(model: Model, fn_img: Path, spell) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')

    spelt = spell(recognized[0])
    print(f'Speller recognises: "{spelt}"')

def main():
    """Main function."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'test', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=15)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
    parser.add_argument('--spell_check_language', help='Either "en" or "ru"', default='en')
    args = parser.parse_args()

    # set chosen CTC decoder
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    spell = Speller(args.spell_check_language)

    # train or validate on IAM dataset
    if args.mode in ['train', 'validate', 'test']:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        char_list = loader.char_list

        # when in line mode, take care to have a whitespace in the char list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        # execute training or validation
        if args.mode == 'train':
            model = Model(char_list, decoder_type)
            train(model, loader, line_mode=args.line_mode, spell=spell, early_stopping=args.early_stopping)
        elif args.mode == 'validate':
            model = Model(char_list, decoder_type, must_restore=True)
            validate(model, loader, args.line_mode, spell)
        elif args.mode == 'test':
            model = Model(char_list, decoder_type, must_restore=True)
            test(model, loader, args.line_mode, spell)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file, spell)


if __name__ == '__main__':
    print("STARTED")
    print(datetime.datetime.now())
    main()
    print("ENDED")
    print(datetime.datetime.now())
