import argparse
import random
from src.wikiscraper import WikiScraper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_topic",
        type=str,
        required=True,
        help="Article name of starting point for crawl",
    )

    parser.add_argument(
        "--n_depth",
        default=2,
        type=int,
        help="Number of recursions in crawl",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Number of recursions in crawl",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    scraper = WikiScraper('data/wiki')
    scraper.collect_wiki_data(['/wiki/{}'.format(args.start_topic)], max_num=args.n_depth)
    scraper.save_wiki_data()

if __name__ == "__main__":
    main()