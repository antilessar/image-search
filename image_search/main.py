from PIL import Image
from sentence_transformers import SentenceTransformer
import vecs
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import argparse

DB_CONNECTION = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"

def seed():
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)

    # create a collection of vectors with 3 dimensions
    images = vx.get_or_create_collection(name="image_vectors", dimension=512)

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')

    # Encode an image:
    img_emb1 = model.encode(Image.open('./images/one.jpeg'))
    img_emb2 = model.encode(Image.open('./images/two.jpeg'))
    img_emb3 = model.encode(Image.open('./images/three.jpeg'))
    img_emb4 = model.encode(Image.open('./images/four.jpeg'))

    # add records to the *images* collection
    images.upsert(
        records=[
            (
                "one.jpeg",        # the vector's identifier
                img_emb1,          # the vector. list or np.array
                {"type": "jpeg"}   # associated  metadata
            ), (
                "two.jpeg",
                img_emb2,
                {"type": "jpeg"}
            ), (
                "three.jpeg",
                img_emb3,
                {"type": "jpeg"}
            ), (
                "four.jpeg",
                img_emb4,
                {"type": "jpeg"}
            )
        ]
    )
    print("Inserted images")

    # index the collection for fast search performance
    images.create_index()
    print("Created index")

def search(search_query):
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)
    images = vx.get_or_create_collection(name="image_vectors", dimension=512)

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')
    # Encode text query
    query_string = search_query
    text_emb = model.encode(query_string)

    # query the collection filtering metadata for "type" = "jpeg"
    results = images.query(
        data=text_emb,                      # required
        limit=1,                            # number of records to return
        filters={"type": {"$eq": "jpeg"}},   # metadata filters
    )
    result = results[0]
    print(result)
    plt.title(result)
    image = mpimg.imread('./images/' + result)
    plt.imshow(image)
    plt.show()


def search_command():
    parser = argparse.ArgumentParser(description='Search images with a prompt.')
    parser.add_argument('--query', type=str, required=True, help='Search query for the search command')
    args = parser.parse_args()
    search(args.query)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search images with a prompt.')
    parser.add_argument('command', choices=['seed', 'search'], help='Command to run (seed or search)')
    parser.add_argument('--query', type=str, help='Search query for the search command')

    args = parser.parse_args()

    if args.command == 'seed':
        seed()
    elif args.command == 'search':
        if not args.query:
            print("Please provide a search query with the --query option.")
        else:
            search(args.query)