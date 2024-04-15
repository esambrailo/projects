'''This module contains the User and Auction classes for the second price auction game.'''
import numpy as np


class User:
    '''Class for Users of the Auction Game. Each user has a secret probability of clicking,
    whenever it is shown an ad. The probability remains constant from creation of the user.
    Probability is drawn from a uniform distribution from 0 to 1.'''
    def __init__(self, user_id=0):
        self.__probability = np.random.uniform()
        self.user_id = user_id

    def __repr__(self):
        return f'{self.user_id}'

    def show_ad(self):
        '''Represents showing an add to this User. Returns True to represent a user clicked,
        and False otherwise.'''
        clicked = np.random.choice([True, False], p=[self.__probability, 1 - self.__probability])
        return clicked

class Auction:
    '''Game involving bidders and users. Each round represents an event in which a user navigates
    to a website with a space for an ad. When this happens bidders place bids, and the winner
    gets to show their ad.  The users may or may not click on the add.  This is a second price
    sealed-bid Auction.

    users: contains a list of all User Objects.
    bidders: is expected to contain a list of all bidder objects.'''

    def __init__(self, users, bidders):
        self.users = users
        self.bidders = bidders
        self.balances = {bidder:0 for bidder in self.bidders}

    def execute_round(self):
        '''All steps within a single round of the game.'''
        #Select random user with uniform probability by referencing user_id
        selected_user = np.random.randint(0, len(self.users))

        #notifying all bidders of the selected user and retreiving their bid amounts.
        bids = []
        for i in range(len(self.bidders)):
            bids.append(self.bidders[i].bid(selected_user))

        #finding the highest bidder and second price
        highest_bid = sorted(bids)[-1]
        if len(bids) == 1:
            second_highest_bid = highest_bid
        else:
            second_highest_bid = sorted(bids)[-2]

        if bids.count(highest_bid) == 1: #checking for ties in the highest price
            winning_bidder = self.bidders[bids.index(highest_bid)]
        else:
            #in the instance of a tie, a new list is created of tied bidders' indexes from
            #which one is selected at random.
            tied = []
            for i in range(len(bids)):
                if bids[i] == highest_bid:
                    tied.append(i)
            winning_bidder = self.bidders[np.random.choice(tied)]

        #running add with user and recording if they click
        clicked = self.users[selected_user].show_ad()

        #notifying the bidders of the results and updating the balances.
        for i in range(len(self.bidders)):
            if self.bidders[i] == winning_bidder:
                self.bidders[i].notify(True, second_highest_bid, clicked)
                self.balances[self.bidders[i]] += (int(clicked) - second_highest_bid)
            else:
                self.bidders[i].notify(False, second_highest_bid, None)
