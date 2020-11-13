import sys
import tweepy
from random import uniform, randint

API_KEY = "ZqQKp99039OZRm3UHAbhkcqwC"
API_SECRET = "SDPCn4nSeeyygKM8egVNkH1P1WsRWD6KlQ6rfX0OsNWBrDFPhJ"
ACCESS_TOKEN = "3402071777-PSWkQn41zK6FNZkXzM09F4nmPkwIllouK3hWQON"
ACCESS_SECRET = "zIv1r5BXb0EOX6Vpa3zWbraU7vPHFhyPb8BZHEYeWxIfB"
HASHTAGS = "hashtags"
TEXT = "text"
FILTER = ["LeagueOfLegends", "Google", "Konami", "Nintendo", "Pokemon", "Trump", "Biden", "Anna", "CAPCOM", "Pokimane", "PewDiePie", "Hentai","Monster Hunter Rise"]
SAMPLE_SIZE = 100

class TwitterStream(tweepy.StreamListener):
    def __init__(self, output_path, sample_size):
        tweepy.StreamListener.__init__(self)
        self.output_path = output_path
        self.sample_size = sample_size
        self.seq_num = 0
        self.total_tags = 0
        self.tags_freq = {}
        self.cache = []
        
        with open(output_path, "w") as f:
            f.close()
        
    
    def on_status(self, status):
        tags = status.entities.get(HASHTAGS)
        tags = [t[TEXT] for t in tags]
        flag = self.process_tags(tags)
        if flag:
            self.write_log()
    
    def increase_seq_num(self):
        self.seq_num += 1
    
    def increase_tags_num(self):
        self.total_tags += 1

    def insert_tag(self, tag):
        if self.sample_size < self.total_tags:
            discarded_pos = self.get_discarded_tag_pos()
            discarded_tag = self.cache[discarded_pos]
            self.tags_freq[discarded_tag] -= 1
            if self.tags_freq[discarded_tag] == 0:
                del self.tags_freq[discarded_tag]
            self.cache[discarded_pos] = tag
        else:
            self.cache.append(tag)
        if tag not in self.tags_freq:
                self.tags_freq[tag] = 0
        self.tags_freq[tag] += 1
        
    def process_tags(self, tags):
        if not tags: return False
        self.increase_seq_num()
        while tags:
            self.increase_tags_num()
            if self.do_discard():
                tags.pop()
            else:
                self.insert_tag(tags.pop())
        return True
    def do_discard(self):
        return uniform(0, 1) > (self.sample_size / self.total_tags)
    
    def get_discarded_tag_pos(self):
        return randint(0, self.sample_size - 1)
    
    def write_log(self):
        tags_arr = [(tag, self.tags_freq[tag]) for tag in self.tags_freq]
        tags_arr.sort(key = lambda x : (-x[1], x[0]))
        with open(self.output_path, "a") as f:
            f.write("The number of tweets with tags from the beginning: {}\n".format(str(self.seq_num)))
            for tag, freq in tags_arr:
                f.write("{} : {}\n".format(tag, str(freq)))
            f.write("\n")
            

        


if __name__ == "__main__":

    argv = sys.argv
    output_path = argv[2]

    ts = TwitterStream(output_path, SAMPLE_SIZE)

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)

    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    stream_obj = tweepy.Stream(auth=auth, listener=ts)
    
    stream_obj.filter(track=FILTER, languages=["en"])