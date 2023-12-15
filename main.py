import fastbook
from fastai.vision.widgets import ImageClassifierCleaner
from fastbook import *
import matplotlib.pyplot as pl

azure_key = os.environ.get('AZURE_SEARCH_KEY', 'ef46f0c9d255472b95608666b8288a34')

def test_azure_key():
    results = search_images_bing(azure_key, 'grizzly bear')
    ims = results.attrgot('contentUrl')

    # print length of results to console
    print(len(ims))

    d = 'grizzly.jpg'
    download_url(ims[0], d)
    image = Image.open(d)
    resized_image = image.to_thumb(128, 128)
    resized_image.show()

def get_images(path_name, types):
    path = Path(path_name)
    if not path.exists():
        path.mkdir()

    for o in types:
        dest = (path / o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azure_key, f'{o} {path_name}')
        download_images(dest, urls=results.attrgot('contentUrl'), max_pics=240,)

    fns = get_image_files(path)
    print(fns)

    # remove any images that can't be opened
    failed = verify_images(fns)
    print(failed)
    failed.map(Path.unlink)

def train_model(path):
    mushrooms = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    dls = mushrooms.dataloaders(path)


    mushrooms = mushrooms.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

    dls = mushrooms.dataloaders(path, num_workers=0) # num_workers=0 to avoid a warning on windows

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    learn.export()

    return learn



def examine_model(model):
    interp = ClassificationInterpretation.from_learner(model)
    interp.plot_confusion_matrix()


    interp.plot_top_losses(5, nrows=1)

    time.sleep(10)


def predict_image(img):
    learn_inf = load_learner('export.pkl')

    img = PILImage.create(img)
    pred,pred_idx,probs = learn_inf.predict(img)

    img.show()

    print(f'Prediction: {pred}; Probability: {probs[pred_idx]*100:.04f}%')



def main():
    # mushroom_types = ['Sutorius eximius']
    # get_images('images', mushroom_types)
    # remove_png_files('images/chanterelle')
    # remove_png_files('images/jack o\' lantern')
    # model = train_model('images')
    # examine_model(model)

    predict_image('chanterelle.jpg')
    predict_image('jol.jpg')
    predict_image('kingb.jpeg')
    predict_image('ShiitakeMushroomsHeader.jpg')
    predict_image('nameko.png')


if __name__ == "__main__":
    main()