import pprint
import socket, ssl, pprint, getpass
import random
import math
from abc import ABC, abstractmethod
import ast

class BasicClientActorAbs(ABC):

    def __init__(self, IP_address = None,verbose=True):
        self.verbose = verbose
        if IP_address == None:
            self.IP_address = '129.241.113.109'
        else:
            self.IP_address = IP_address

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # We require a certificate from the server. We used a self-signed certificate
        # so here ca_certs must be the server certificate itself.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations("server.crt")
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False # We have no hostname for the server
        self.ssl_sock = context.wrap_socket(self.s)

        self.series_id = -1



    def connect_to_server(self):
        """
        We establish an encrypted connection to the server, using the IP-address and port number specified by the IT3105
        staff. This will start a process where you will be asked to enter your NTNU credentials and a player-name for
        your actor. When the correct credentials are entered and verified by the server, a tournament will automatically
        be started by the server.

        ------------------------------------------ IMPORTANT NOTES ------------------------------------------------------
        If you decide to automate the process of entering your user credentials, you should NOT store your password in
        the code. You can store it in a separate file, but you must then make sure that this file is not uploaded to
        either Blackboard or any version control system you are using. Failure to do so might compromise your NTNU
        student account.

        This server ONLY responds to requests from computers on the NTNU network.  Therefore you either need to be
        on campus or use VPN to have access.

        ------------------------------------------ IMPORTANT NOTES ------------------------------------------------------

        :return:
        """
        # Tell us to what address we are connecting
        print('Attempting to connect to server using ip-address: ' +self.IP_address+':33000...')

        # to work. Do this either by being at the campus or by using a VPN.
        self.ssl_sock.connect((self.IP_address, 33000))


        # Print certificate
        pprint.pprint(self.ssl_sock.getpeercert())

        # Print the cipher used for encrypted connection
        print(self.ssl_sock.cipher())

        # Receive initial login dialog
        while True:
            msg = self.ssl_sock.recv(1024).decode('utf8')

            # We are asked to enter our NTNU username
            if "username" in msg:
                usr_in = input(msg)

            # We are asked to enter our NTNU password
            elif "password" in msg:
                usr_in = getpass.getpass(msg)

            # If we are successful the server will tell us and we can start a game!
            elif "Welcome" in msg:
                print(msg)

                # Start playing tournament
                self.play_tournament()
                # Tournament finished, disconnect from server.
                self.disconnect_from_server()


            # We entered wrong credentials
            elif "Invalid credentials" in msg:
                print("Wrong credentials")
                print(msg)
                self.disconnect_from_server()

            # We are asked to enter the name we want for our player in the tournament
            elif "player-name" in msg:
                usr_in = input(msg)

            # We are asked if we want to qualify or play a test game
            elif "qualify" in msg:
                usr_in = input(msg)

            # We are asked if we want to qualify or play a test game
            elif "stress" in msg:
                usr_in = input(msg)

            # We have no more qualification attempts remaining
            elif "Sorry" in msg:
                exit()

            elif "User did not want to participate in stress test." in msg:
                exit()

            # Unrecognized server response
            else:
                print('Unrecognized response from server, disconnecting now.')
                self.disconnect_from_server()

            # Send user response to server.
            self.ssl_sock.send(bytes(usr_in, 'utf8'))

    def show_state(self,state):
        if self.verbose:
            print(state)

    def play_tournament(self):
        """
        This is the main loop of an actor-server connection. It plays a tournament against the server and will receive
        a final score for your actor in the end. A tournament consists of three processes.

        - A game: A game is a single round of HEX played against one of the players on the server.

        - A series: A series consists of several games against the same player on the server. This is to ensure that a
                    fair score against this player is calculated.

        - A tournament: A tournament consists of all the series played against a collection of players on the server.
                        This will result in a final score for your actor, based on how well it did in the tournament.

        When a tournament is played the actor will receive messages from the server and generate responses based on
        what these message are. There are SEVEN possible messages:

        - Series start: A new series is starting and your actor is given several pieces of information.
        - Game start: A new game is starting, and you are informed of which player will make the first move.
        - Game end: The current game has ended and you receive basic information about the result.
        - Series end: A series has ended and you will receive basic summarizing statistics.
        - Tournament end; A tournament has ended. When a tournament is ended, you will receive your final score. You
                          should save this score and verify that it is the same score you are given in our system at the
                          end of the server-client week.
        - Illegal action: The previous action your actor tried to execute was evaluated to be an illegal action by the
                          server. The server sends you the previous state and the attempted action by your actor.
                          Finally the server will close down the connection from its end. You should analye the
                          previous state and your attempted action and try to find the bug in your action generator.
        - Request action: The server will request actions from your actor. You will receive a state tuple, representing
                          the current state of the board, and you will use your actor to find the next move. This move
                          will then be sent to the server; it must be a move tuple, (row, col), which represent the next
                          empty cell in which your actor wants to place a piece.

        :return:
        """
        while True:
            # Receive a message from the server
            state = self.ssl_sock.recv(1024).decode('utf8')
            self.show_state(state)
            # We received a series start message.
            if state == 'Series start':
                unique_player_id = ast.literal_eval(self.ssl_sock.recv(1024).decode('utf8'))
                player_id_map = ast.literal_eval(self.ssl_sock.recv(1024).decode('utf8'))
                num_games = ast.literal_eval(self.ssl_sock.recv(1024).decode('utf8'))
                game_params = ast.literal_eval(self.ssl_sock.recv(1024).decode('utf8'))
                series_player_id = [p[1] for p in player_id_map if p[0] == unique_player_id][0]
                self.handle_series_start(unique_player_id,series_player_id,player_id_map,num_games,game_params)

            elif state == 'Game start':
                start_player = self.ssl_sock.recv(1024).decode('utf8')
                self.handle_game_start(ast.literal_eval(start_player))

            # We received a game end message
            elif state == 'Game end':
                winner = self.ssl_sock.recv(1024).decode('utf8')
                end_state = self.ssl_sock.recv(1024).decode('utf8')
                self.handle_game_over(ast.literal_eval(winner), ast.literal_eval(end_state))

            # We received a series end message
            elif state == 'Series end':
                stats = self.ssl_sock.recv(1024).decode('utf8')
                self.handle_series_over(ast.literal_eval(stats))

            # We received a tournament end message
            elif state == 'Tournament end':
                score = self.ssl_sock.recv(1024).decode('utf8')
                self.handle_tournament_over(ast.literal_eval(score))
                break

            # We received an illegal action message
            elif state == 'Illegal action':
                state = self.ssl_sock.recv(1024).decode('utf8')
                illegal_action = self.ssl_sock.recv(1024).decode('utf8')
                self.handle_illegal_action(ast.literal_eval(state), ast.literal_eval(illegal_action))
                break

            # We received a request for a new move
            else:
                usr_move = str(self.handle_get_action(ast.literal_eval(state)))
                self.ssl_sock.send(bytes(usr_move, 'utf8'))

    @abstractmethod
    def handle_get_action(self, state):
        """
        Here you will use your DeepLearning-MCTS to select a move for your actor on the current board. Remember to user
        the correct player_number for YOUR actor! The default action is to select a random empty cell on the board.
        This should be modified.
        :param state: The current board
        :return:
        """
        pass


    @abstractmethod
    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return
        """
        pass

    @abstractmethod
    def handle_game_start(self, start_player):
        """
        :param start_player: The player number (1 or 2) who will start this particular game
        :return
        """
        pass

    @abstractmethod
    def handle_game_over(self, winner, end_state):
        """
        Here you can decide to handle what happens when a game finishes. The default aciton is to print the winner and
        the end state.
        :param winner: Winner ID
        :param end_state: Final state of the board.
        :return:
        """
        pass

    @abstractmethod
    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want, the initial handling just prints the stats.
        :param stats: The actor statistics for a series
        :return:
        """
        pass

    @abstractmethod
    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        pass

    @abstractmethod
    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        pass

    def disconnect_from_server(self):
        """
        This method closes the connection to the server
        :return:
        """
        self.ssl_sock.close()
        exit()

    def pick_random_free_cell(self, state, size):
        """
        This method selects a random move, based on which cells are free on the board. If you have aims to win the
        tournament, this should not be your default move ;)
        :param size: The size of the board
        :return: random move
        """
        empty_locs = []
        for index, item in enumerate(state[1:]):
            if item == 0:
                row = math.floor(index / size)
                col = index % size
                empty_locs.append((row, col))
        return random.choice(empty_locs)




