# coding = utf-8

import random
from operator import itemgetter
import csv
import pickle
from pyspark import SparkContext
import os
from pyspark.sql import SparkSession
import json
from hdfs import InsecureClient
from hdfs.util import HdfsError


class ItemBasedCF():
    # Initializing parameters
    def __init__(self):
        # Find 20 similar movies and recommend 10 movies for the target user
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # The dataset is divided into training set and test set
        self.trainSet = {}
        self.testSet = {}

        # User similarity matrix
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        # Set up HDFS connection information
        self.hdfs_host = 'http://master:9870'  # HDFS host address and port
        self.hdfs_user = 'hadoop'  # HDFS username

        # print('Similar movie number = %d' % self.n_sim_movie)
        # print('Recommneded movie number = %d' % self.n_rec_movie)


    ## Read the file to get the "movies-actor-production team" data
    # def get_dataset(self, filename, pivot=0.75):
    #     trainSet_len = 0
    #     testSet_len = 0
    #     with open(filename, 'r', encoding='utf-8') as csvfile:
    #         csvreader = csv.reader(csvfile)
    #         next(csvreader)  # Skip the header line
    #         for row in csvreader:
    #             movie_id, title, cast, crew = row
    #             if random.random() < pivot:
    #                 self.trainSet.setdefault(movie_id, {})
    #                 self.trainSet[movie_id]['title'] = title
    #                 self.trainSet[movie_id]['cast'] = cast.split(',')
    #                 self.trainSet[movie_id]['crew'] = crew.split(',')
    #                 trainSet_len += 1
    #             else:
    #                 self.testSet.setdefault(movie_id, {})
    #                 self.testSet[movie_id]['title'] = title
    #                 self.testSet[movie_id]['cast'] = cast.split(',')
    #                 self.testSet[movie_id]['crew'] = crew.split(',')
    #                 testSet_len += 1
    #     print('Split trainingSet and testSet success!')
    #     print('TrainSet = %s' % trainSet_len)
    #     print('TestSet = %s' % testSet_len)

    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        with open(filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header line
            for row in csvreader:
                movie_id, title, cast_str, crew_str = row
                cast = cast_str.split(',') 
                crew = crew_str.split(',') 

                if random.random() < pivot:
                    self.trainSet.setdefault(movie_id, {})
                    self.trainSet[movie_id]['title'] = title
                    self.trainSet[movie_id]['cast'] = cast
                    self.trainSet[movie_id]['crew'] = crew
                    trainSet_len += 1
                else:
                    self.testSet.setdefault(movie_id, {})
                    self.testSet[movie_id]['title'] = title
                    self.testSet[movie_id]['cast'] = cast
                    self.testSet[movie_id]['crew'] = crew
                    testSet_len += 1

        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    # Uploading files
    def save_csv_to_hdfs(self, filename):
        # Create an HDFS client
        client = InsecureClient(self.hdfs_host, user=self.hdfs_user)



        # Upload local files to HDFS
        hdfs_file_path = filename
        try:
            # Checks if the file exists
            client.status(hdfs_file_path)

            # If the file exists, it is skipped
            print(f"file '{hdfs_file_path}' Already exists, skip upload.")

        except HdfsError:
            # If the file does not exist, it is uploaded
            client.upload(hdfs_file_path, "E:\\ITM COURSES\\7095\\gpj\\" + filename)
            print(f"file '{hdfs_file_path}' Uploading successfully.")

    # Downloading the file
    def download_to_hdfs(self, filename):
        # Create an HDFS client
        client = InsecureClient(self.hdfs_host, user=self.hdfs_user)

        hdfs_file_path =  filename
        # Download the HDFS file locally
        filename_new = "E:\\ITM COURSES\\7095\\gpj\\" + filename.split(".")[0]+"_new.csv"
        client.download(hdfs_file_path, filename_new)



    # Read the file and return each line of the file
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # Remove the title from the first line of the file
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    # Calculate the similarity between movies (taking into account cast and production team information)
    def calc_movie_sim(self):
        for movie1 in self.trainSet:
            for movie2 in self.trainSet:
                if movie1 == movie2:
                    continue
                common_cast = set(self.trainSet[movie1]['cast']) & set(self.trainSet[movie2]['cast'])
                common_crew = set(self.trainSet[movie1]['crew']) & set(self.trainSet[movie2]['crew'])
                similarity = len(common_cast) + len(common_crew)
                self.movie_sim_matrix.setdefault(movie1, {})
                self.movie_sim_matrix[movie1].setdefault(movie2, 0)
                self.movie_sim_matrix[movie1][movie2] = similarity
        print('Calculate movie similarity matrix success!')

    def calc_movie_sim_spark(self):
        sc = SparkContext.getOrCreate()

        # Creating an RDD
        movie_rdd = sc.parallelize(list(self.trainSet.items()))

        # Calculate the similarity matrix
        movie_sim_matrix = movie_rdd.cartesian(movie_rdd) \
            .filter(lambda x: x[0][0] != x[1][0]) \
            .map(lambda x: self.calc_sim(x)) \
            .collectAsMap()

        self.movie_sim_matrix = movie_sim_matrix

    # The helper function calculates the similarity between two movies
    def calc_sim(self, x):
        movie1, movie1_info = x[0]
        movie2, movie2_info = x[1]

        cast1 = movie1_info['cast']
        crew1 = movie1_info['crew']
        cast2 = movie2_info['cast']
        crew2 = movie2_info['crew']

        common_cast = len(set(cast1) & set(cast2))
        common_crew = len(set(crew1) & set(crew2))

        sim = common_cast + common_crew

        return (movie1, movie2), sim

    # Recommend similar movies (considering casting and production team information)
    def recommend(self, movie_titles):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_cast = set()
        watched_crew = set()

        for title in movie_titles:
            for movie_id, movie_info in self.trainSet.items():
                if movie_info['title'] == title:
                    watched_cast.update(movie_info['cast'])
                    watched_crew.update(movie_info['crew'])
                    break

        for movie_id, movie_info in self.trainSet.items():
            if movie_info['title'] in movie_titles:
                continue
            movie_cast = set(movie_info['cast'])
            movie_crew = set(movie_info['crew'])
            common_cast = len(watched_cast & movie_cast)
            common_crew = len(watched_crew & movie_crew)
            total_similarity = common_cast + common_crew
            rank[movie_info['title']] = total_similarity

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]



    # Recommendations are generated and evaluated by precision, recall, and coverage
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie
        # Precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # Rate of coverage
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            if not test_movies:
                continue
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        if test_count == 0:
            print("No movies in test set for recommendations.")
            return

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

    # Saving the model
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    # Loading the model
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model



if __name__ == '__main__':
    rating_file = 'tmdb_5000_credits.csv'
    hdfs_target_path = ""
    itemCF = ItemBasedCF()
    # itemCF.get_dataset("E:\\ITM COURSES\\7095\\gpj\\"+rating_file)
    # itemCF.calc_movie_sim()
    # # itemCF.calc_movie_sim_spark() # Compute the similarity matrix using Spark

    # # Saving the model
    # itemCF.save_model('E:\\ITM COURSES\\7095\\gpj\\itemCF_model.pkl')
    # # Save the CSV data to HDFS
    # itemCF.save_csv_to_hdfs(rating_file)
    # itemCF.download_to_hdfs( rating_file)

    # itemCF.evaluate()



    # Loading the model
    loaded_model = ItemBasedCF.load_model('E:\\ITM COURSES\\7095\\gpj\\itemCF_model.pkl')



    # A sequence of movie titles that the user likes
    # Each movie is separated by a comma such as Tangled, Avengers: Age of Ultron, Avatar
    user_input = input("Please enter movies: \n")
    user_likes =user_input.split(",")
    # user_likes = ['Tangled', 'Avengers: Age of Ultron', 'Avatar']
    recommendations = loaded_model.recommend(user_likes)
    # print(recommendations)

    # # Converts the movie ID to a movie name
    # movie_id_to_title = {movie_info['title']: movie_id for movie_id, movie_info in itemCF.trainSet.items()}
    # recommendations_with_titles = [(movie_id_to_title.get(movie_id, 'Unknown'), score) for movie_id, score in
    #                                recommendations]

    print("Movies recommended to the user: \n", recommendations)
