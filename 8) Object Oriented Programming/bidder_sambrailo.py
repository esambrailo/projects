'''This module contains the Bidder class for the second price auction game.'''

class Bidder:
    '''Class for Bidders of Auction game. Each bidder begins with a balance of
    0 dollars. Object of the bidder is to finish the Auction game with as high
    a balance as possible.  Negative balances are okay, until the balance reaches
    -1000 dollars, which will result in disqualification.

    num_users: number of User objects in the game.
    num_rounds: contains the total number of rounds to be played.'''

    def __init__(self, num_users, num_rounds):
        self.num_users = num_users
        self.num_rounds = num_rounds
        #user.data is a dictionary storing various data for each user.
        #The values per each user are:
            #[index 0: 'times user is drawn',
            # index 1: 'times Bidder has won bid with this user',
            # index 2: 'accrued cost of winning bids for this user,
            # index 3: 'total winnings for Bidder for this user',
            # index 4: 'what Bidder's bid amount will be next time user is drawn.]
        self.user_data = {user:[0,0,0,0,.50] for user in range(self.num_users)}
        self.current_user_id = int()

    def bid(self, user_id):
        '''Returns a non-negative amount of money, in dollars round to (3) decimal places.'''
        self.current_user_id = user_id
        bid_amount = round(self.user_data[self.current_user_id][4], 3)
        return bid_amount

    def notify(self, auction_winner, price, clicked):
        '''Used to receive information about what happened in a round from the Auction.

        auction_winner: boolean to represent whether the given Bidder won the auction.
        price: the amount of the second bid, which the winner pays.
        clicked: will contain a boolean value to represent whether the user clicked on
        the add. None if bidder did not win auction.'''

        #using notify to update user_data with pertinent information,
        #and modify the bid amount Bidder will use next time user is drawn
        self.user_data[self.current_user_id][0] += 1 #adding to user draw count
        #below adding to Bidder's win count with user
        self.user_data[self.current_user_id][1] += int(auction_winner)
        #below adding to accrued cost of winning bids
        self.user_data[self.current_user_id][2] += price
        if auction_winner: #adding to winnings for Bidder if user clicked
            self.user_data[self.current_user_id][3] += int(clicked)

        #below is strategy for amount Bidder will bid next time user is drawn
        if self.user_data[self.current_user_id][1] == 0:
            self.user_data[self.current_user_id][4] = 1 #having initial bid of 1 for all users.
        #for the first 10% of probable bids for a respective bidder, where Bidder won,
        # but user did not click, the bid amount is calcuated as (1/('# of bids won'*2))
        elif self.user_data[self.current_user_id][1] < (self.num_rounds/self.num_users)/10 and self.user_data[self.current_user_id][3] == 0:
            self.user_data[self.current_user_id][4] = 1/(self.user_data[self.current_user_id][1]*2)
        #beyond that, the bid is the percentage of user clicks to total bid won.
        else:
            self.user_data[self.current_user_id][4] = self.user_data[self.current_user_id][3]/self.user_data[self.current_user_id][1]
