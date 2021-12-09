import csv
import os

def data_preprocess():
    def get_data(filename, dataname):
        neg_texts = []
        hc_pos_texts = []
        ac_pos_texts = []

        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                if row[0] == dataname and row[1] == "neg":
                    neg_texts.append(row[-1])
                if row[0] == dataname and row[1] == "pos" and row[3]=="hc5" :
                    hc_pos_texts.append(row[-1])
                if row[0] == dataname and row[1] == "pos" and row[3]=="ac" :
                    ac_pos_texts.append(row[-1])
                i += 1

        new_neg_texts, new_hc_pos_texts, new_ac_pos_texts = [], [], []
        for text in neg_texts:
            if text not in new_neg_texts:
                new_neg_texts.append(text)
        for text in hc_pos_texts:
            if text not in new_hc_pos_texts:
                new_hc_pos_texts.append(text)
        for text in ac_pos_texts:
            if text not in new_ac_pos_texts:
                new_ac_pos_texts.append(text)
        return [new_neg_texts,new_hc_pos_texts,new_ac_pos_texts]

    def write_csv(data_0,data_1, filename):
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text","label"])
            for sent in data_0:
                writer.writerow([sent,0])
            for sent in data_1:
                writer.writerow([sent,1])

    csv_file ="./steganalysis_dataset_2.csv"
    movie = get_data(csv_file,dataname="movie")
    news = get_data(csv_file,dataname="news")
    tweet = get_data(csv_file, dataname="tweet")

    DIR = "./vae_raw_data"
    os.makedirs(DIR,exist_ok=True)
    movie_dir = os.path.join(DIR, "movie")
    news_dir = os.path.join(DIR, "news")
    tweet_dir = os.path.join(DIR, "tweet")
    os.makedirs(movie_dir, exist_ok=True)
    os.makedirs(news_dir, exist_ok=True)
    os.makedirs(tweet_dir,exist_ok=True)
    write_csv(movie[0][:10000], movie[1][:10000], os.path.join(movie_dir, "movie_0_1_hc.csv"))
    write_csv(movie[0][:10000], movie[2][:10000], os.path.join(movie_dir, "movie_0_1_ac.csv"))
    write_csv(news[0][:10000], news[1][:10000], os.path.join(news_dir, "news_0_1_hc.csv"))
    write_csv(news[0][:10000], news[2][:10000] , os.path.join(news_dir, "news_0_1_ac.csv"))
    write_csv(tweet[0][:10000], tweet[1][:10000] , os.path.join(tweet_dir, "tweet_0_1_hc.csv"))
    write_csv(tweet[0][:10000],tweet[2][:10000] , os.path.join(tweet_dir, "tweet_0_1_ac.csv"))

if __name__ == '__main__':
    data_preprocess()
