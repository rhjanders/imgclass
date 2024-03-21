import argparse
import os
import sys
from duckduckgo_search import DDGS
from fastcore.all import *
from time import sleep
from fastdownload import download_url
from fastai.vision.all import *

# 1. Test data download part: hand-picked URLs of jets to be classified - image copyright: authors :)
urls = {
        "sr71-burners-side": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.pinimg.com%2Foriginals%2Fe2%2F69%2F29%2Fe269291dc68c868a124e24e8c5108a52.jpg&f=1&nofb=1&ipt=03436b4c34edd6f3d905d021547bdb68c69b7fbe7ffc320091249e5b3b2ae2e0&ipo=images",
        "sr71-burners-below": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F64.media.tumblr.com%2F15605183476d6d4485f36dca6e838f22%2Ftumblr_oh0t2yXhwi1thnmmgo1_1280.jpg&f=1&nofb=1&ipt=dd168de039b5bf8a34689383b2c5b509265d9c1feda3b717f0b25c1ada3d5194&ipo=images",
        "sr71-front": "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F4.bp.blogspot.com%2F-zw-Z_b9ia5c%2FURVjpgc-M6I%2FAAAAAAAAgKk%2FXlDkNP-KSvg%2Fs1600%2FLockheed_SR-71_Blackbird_2.jpg&f=1&nofb=1&ipt=459814548a081a517d06e829a0cd3364ea4dee7bcb69e84dba452737726cda00&ipo=images",
        "sr71-side": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.j-AtYMtPurO3re3P7vdb5wHaEo%26pid%3DApi&f=1&ipt=4efbe3189d28f555948d8aff46ad8bb4c6eea0eba50b3643ecf2f279ab252ade&ipo=images",
        "reference-f14": "https://upload.wikimedia.org/wikipedia/commons/f/f7/US_Navy_051105-F-5480T-005_An_F-14D_Tomcat_conducts_a_mission_over_the_Persian_Gulf-region.jpg",
        "tiger": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Flookaside.fbsbx.com%2Flookaside%2Fcrawler%2Fmedia%2F%3Fmedia_id%3D100066518556292&f=1&nofb=1&ipt=a15df7fe58ff3522d423ebf40e559c472015eeb2896f2e17ff2f2ef38e1cb9da&ipo=images",
        "lion": "https://hdqwalls.com/wallpapers/lion-4k.jpg",
        "durian": "https://www.treehugger.com/thmb/lmPJaHKyddBuIZuPRUJZQITSRmE=/4256x2832/filters:fill(auto,1)/__opt__aboutcom__coeus__resources__content_migration__mnn__images__2018__12__durian-ripe-sliced-table-aee2df44cfff4c088884ead337f15205.jpg",
        "jackfruit": "https://images.slurrp.com/prod/articles/2j9iob2ji96.webp?impolicy=slurrp-20210601&width=1200&height=900&q=75",
        "rocket": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.arstechnica.net%2Fwp-content%2Fuploads%2F2018%2F10%2FNewGlenn-2.jpeg&f=1&nofb=1&ipt=58ec502a986f721e49ee927318e0a5a65e150136683db51c27d75daebce4e75d&ipo=images"
        }

# create working dir
dir = "data/test"
path = Path(dir)
path.mkdir(exist_ok=True, parents=True)

# download the images
for key, value in urls.items():
    path = dir + "/" + key + ".jpg"
    urlretrieve(value, path)
    print("Downloaded file ", path)

# resize them to match the requirements of the classifier
resize_images(dir, max_size=400, dest=dir)

# remove failed downloads
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

# 2. learning data download part

# CLI argument parser

parser = argparse.ArgumentParser()
parser.add_argument("--first", help="First image category to search",
                    default="jetfighter")
parser.add_argument("--second", help="Second image category to search",
                    default="rocket")
args = vars(parser.parse_args())

# end parser

# wrapper method around duckduckgo image search
def search_images(term, max_images=30):
    print(f"Searching for '{term}' and downloading matching images")
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')

# create working dir
searches = args['first'], args['second']
dir = "data" + "/" + args['first'] + "_or_" + args['second']
path = Path(dir)
path.mkdir(exist_ok=True, parents=True)


# download images
for object in searches:
    dest = (path/object)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{object} photo'))
    resize_images(path/object, max_size=400, dest=path/object)

print("Learning images written to ", dir)

# remove failed downloads
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

