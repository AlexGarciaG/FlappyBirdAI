from email import message
import pygame
import neat
import time
import os
import random
import pickle
import flappyClasses
import visualize


#Pygame font
pygame.font.init()
STAT_FONT = pygame.font.SysFont("FreeMono, Monospace",30)
class flappyGame :
    def __init__(self,config_path_file):
        #game
        self.clock = pygame.time.Clock()
        self.PIPES_GAP = 650
        #Window size
        self.WINDOW_WIDTH = 500
        self.WINDOW_HEIGHT = 800
        #Display
        self.win = pygame.display.set_mode((self.WINDOW_WIDTH,self.WINDOW_HEIGHT))
        pygame.display.set_caption('Flappy bird AI')
        #Neat config file
        self.config_path=config_path_file
        # Load requried NEAT config
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path)

        self.gen = 0
        #Back ground
        self.ticks = 30
        self.ticksTrain = 120
        self.birds = []
        self.nets = []
        self.ge   = []
        self.base  = flappyClasses.Base(730)
        self.genomes = []
        #Menu
        self.back_button = (100,100,100)
        self.letter_button = (255,255,255)
        self.button_height = 35
        self.xPadding = 10
        self.yPadding = 0
        self.xOrigin = 9
        self.wightSize = 19
        self.butonGap = 100
            #Play User
        self.message_Play_User = "Play User"
        self.x_Play_User = self.WINDOW_WIDTH/2-(len(self.message_Play_User)*self.xOrigin)-self.xPadding
        self.y_Play_User = 100-self.yPadding
        self.width_Play_User = (len(self.message_Play_User)*self.wightSize)+self.xPadding

            #Train Genetic Algorithm
        self.message_Train_Genetic = "Train Genetic Algorithm"
        self.x_Train_Genetic = self.WINDOW_WIDTH/2-(len(self.message_Train_Genetic)*self.xOrigin)-self.xPadding
        self.y_Train_Genetic = self.y_Play_User-self.yPadding+self.butonGap
        self.width_Train_Genetic = (len(self.message_Train_Genetic)*self.wightSize)+self.xPadding

            #Play Genetic Algorithm
        self.message_Play_Genetic = "Play Genetic Algorithm"
        self.x_Play_Genetic = self.WINDOW_WIDTH/2-(len(self.message_Play_Genetic)*self.xOrigin)-self.xPadding
        self.y_Play_Genetic = self.y_Train_Genetic-self.yPadding+self.butonGap
        self.width_Play_Genetic = (len(self.message_Play_Genetic)*self.wightSize)+self.xPadding

            #Play Neural Network
        self.message_Play_Neural = "Play Neural Network"
        self.x_Play_Neural = self.WINDOW_WIDTH/2-(len(self.message_Play_Neural)*self.xOrigin)-self.xPadding
        self.y_Play_Neural = self.y_Play_Genetic-self.yPadding+self.butonGap
        self.width_Play_Neural = (len(self.message_Play_Neural)*self.wightSize)+self.xPadding

            #Play Logistic Regression
        self.message_Play_Logistic = "Play Logistic Regression"
        self.x_Play_Logistic = self.WINDOW_WIDTH/2-(len(self.message_Play_Logistic)*self.xOrigin)-self.xPadding
        self.y_Play_Logistic = self.y_Play_Neural-self.yPadding+self.butonGap
        self.width_Play_Logistic = (len(self.message_Play_Logistic)*self.wightSize)+self.xPadding

            #Settings
        self.message_Settings = "Settings"
        self.x_Settings = self.WINDOW_WIDTH/2-(len(self.message_Settings)*self.xOrigin)-self.xPadding
        self.y_Settings = self.y_Play_Logistic-self.yPadding+self.butonGap
        self.width_Settings = (len(self.message_Settings)*self.wightSize)+self.xPadding

            #Documentation
        self.message_Documentation = "Documentation"
        self.x_Documentation = self.WINDOW_WIDTH/2-(len(self.message_Documentation)*self.xOrigin)-self.xPadding
        self.y_Documentation = self.y_Settings-self.yPadding+self.butonGap
        self.width_Documentation = (len(self.message_Documentation)*self.wightSize)+self.xPadding



        #Genetic Algorithm
        self.livesPerGen = 5
            #Exit
        self.message_Back_Menu = "Back Menu"
        self.x_Back_Menu = 0
        self.y_Back_Menu = 0
        self.width_Back_Menu = (len(self.message_Back_Menu)*self.wightSize)+self.xPadding
    #Draw Game
    def draw_window_game(self,birds,pipes,base,score):
        self.win.blit(flappyClasses.BG_IMG,(0,0))
        for pipe in pipes:
            pipe.draw(self.win)
        text = STAT_FONT.render("Score:"+str(score),1,(255,255,255))
        self.win.blit(text,(self.WINDOW_WIDTH-10-text.get_width(),10))

        text = STAT_FONT.render("Gen:"+str(self.gen),1,(255,255,255))
        self.win.blit(text,(10,10))

        base.draw(self.win)
        for bird in birds:
            bird.draw(self.win)
        pygame.display.update()
    #Game logic
    def fitness_function (self,genomes,config):
        self.gen +=1
        #Config nat
        nets = []
        ge   = []
        birds = []
        #Create a bird for each gen
        for _,g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g,config)
            nets.append(net)
            birds.append(flappyClasses.Bird(230,350))
            g.fitness  = 0
            ge.append(g)
        #Define basic map
        base = flappyClasses.Base(730)
        pipes = [flappyClasses.Pipe(self.PIPES_GAP)]
        #Define display
        #Define when the game ends
        run = True
        #Define clock game
        clock = pygame.time.Clock()
        #Define score
        score = 0
        while run:
            #clock delay
            clock.tick(self.ticksTrain )
            #Lock if game window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
            #Check near pipe
            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                    pipe_ind = 1    
            else:                                                           
                run = False
                break
            #Move birds
            for x,bird in enumerate(birds):
                bird.move()
                ge[x].fitness   += 0.1
                #get output to move bird
                output = nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].botton)))
                if output[0] > 0.5:
                    bird.jump()


            #Var for detection game changes
            add_pipe = False
            rem = []
            #Check collition on pipes
            for pipe in pipes:
                #cheack collition for each bird
                for x,bird in enumerate(birds):
                    #checj collition
                    if pipe.collide(bird):
                        #Penalty if it heats a pipe
                        ge[x].fitness  -= 1 
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    # If a bird pass a pipe enable add pipe and passed
                    if not pipe.passed and pipe.x <bird.x :
                        pipe.passed = True
                        add_pipe = True
                # If a pipe left scream append a pipe to rem so it can be delate leater
                if pipe.x + pipe.PIPE_TOP.get_width()< 0:
                    rem.append(pipe)
                #Move pipes
                pipe.move()
            #Increase score when a pipe is passed
            if add_pipe:
                score += 1 
                #Give reward to birds that passed pipes
                for g in ge:
                    g.fitness  += 5
                pipes.append(flappyClasses.Pipe(self.PIPES_GAP))
            #Remove pipes
            for r in rem:
                pipes.remove (r)
            #Check collition whit base or exit scream
            for x,bird in enumerate(birds):
                    #check collition
                if bird.y + bird.img.get_height ()>= 730 or bird.y < 0:
                    #Penalty if it heats a pipe
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            if score > 50 :
                break
            base.move()
            #Draw game
            self.draw_window_game( birds,pipes,base,score)
        #If game stop quit
    def run(self):
        self.gen = 0
        #Get config file
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            self.config_path)
        #Pass config to neat
        p = neat.Population(config)
        #Print report to terminal
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        #Execute fitnes function with 50 gens
        winner = p.run (self.fitness_function,self.livesPerGen )
        print('\nBest genome:\n{!s}'.format(winner))
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)
            f.close()
    def draw_window_game_Replay(self,birds,pipes,base,score):
        self.win.blit(flappyClasses.BG_IMG,(0,0))
        for pipe in pipes:
            pipe.draw(self.win)
        text = STAT_FONT.render("Score:"+str(score),1,(255,255,255))
        self.win.blit(text,(self.WINDOW_WIDTH-10-text.get_width(),10))

        base.draw(self.win)
        for bird in birds:
            bird.draw(self.win)
        
        #Back
        pygame.draw.rect(self.win, self.back_button, [self.x_Back_Menu, self.y_Back_Menu ,self.width_Back_Menu , self.button_height])
        text = STAT_FONT.render(self.message_Back_Menu,1,self.letter_button)
        self.win.blit(text,(self.x_Back_Menu+self.xPadding,self.y_Back_Menu+self.yPadding))

        pygame.display.update()
    #Game logic
    def fitness_function_Replay (self,genomes,config):
        #Config nat
        nets = []
        ge   = []
        birds = []
        #Create a bird for each gen
        for _,g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g,config)
            nets.append(net)
            birds.append(flappyClasses.Bird(230,350))
            g.fitness  = 0
            ge.append(g)
        #Define basic map
        base = flappyClasses.Base(730)
        pipes = [flappyClasses.Pipe(self.PIPES_GAP)]
        #Define display
        #Define when the game ends
        run = True
        #Define clock game
        clock = pygame.time.Clock()
        #Define score
        score = 0
        while run:
            #clock delay
            clock.tick(self.ticks )
            #Lock if game window is closed
            for event in pygame.event.get():
                self.mouse = pygame.mouse.get_pos()
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Back_Menu <= self.mouse[0] <= self.x_Back_Menu+self.width_Back_Menu) and (self.y_Back_Menu <= self.mouse[1] <= self.y_Back_Menu+self.button_height):
                        run = False
                        print("Main Menu")
            #Check near pipe
            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                    pipe_ind = 1    
            else:                                                           
                run = False
                break
            #Move birds
            for x,bird in enumerate(birds):
                bird.move()
                ge[x].fitness   += 0.1
                #get output to move bird
                output = nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].botton)))
                if output[0] > 0.5:
                    bird.jump()


            #Var for detection game changes
            add_pipe = False
            rem = []
            #Check collition on pipes
            for pipe in pipes:
                #cheack collition for each bird
                for x,bird in enumerate(birds):
                    #checj collition
                    if pipe.collide(bird):
                        #Penalty if it heats a pipe
                        ge[x].fitness  -= 1 
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    # If a bird pass a pipe enable add pipe and passed
                    if not pipe.passed and pipe.x <bird.x :
                        pipe.passed = True
                        add_pipe = True
                # If a pipe left scream append a pipe to rem so it can be delate leater
                if pipe.x + pipe.PIPE_TOP.get_width()< 0:
                    rem.append(pipe)
                #Move pipes
                pipe.move()
            #Increase score when a pipe is passed
            if add_pipe:
                score += 1 
                #Give reward to birds that passed pipes
                for g in ge:
                    g.fitness  += 5
                pipes.append(flappyClasses.Pipe(self.PIPES_GAP))
            #Remove pipes
            for r in rem:
                pipes.remove (r)
            #Check collition whit base or exit scream
            for x,bird in enumerate(birds):
                    #check collition
                if bird.y + bird.img.get_height ()>= 730 or bird.y < 0:
                    #Penalty if it heats a pipe
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            if score > 50 :
                break
            base.move()
            #Draw game
            self.draw_window_game_Replay( birds,pipes,base,score)
        #If game stop quit
    def replay_genome(self, genome_path="winner.pkl"):
        self.gen = 0
        # Load requried NEAT config
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path)

        # Unpickle saved winner
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        node_names = {-1:'Bird y', -2: 'Dist to pipe top', -3:'Dist to pipe bottom',0:'Jump'}
        visualize.draw_net(config, genome, True,node_names=node_names)
        print('\nBest genome:\n{!s}'.format(genome))
        # Convert loaded genome into required data structure
        genomes = [(1, genome)]

        # Call game with only the loaded genome
        self.fitness_function_Replay(genomes, config)
    
    def draw_window_menu(self):
        self.win.blit(flappyClasses.BG_IMG,(0,0))
        for pipe in self.pipes:
            pipe.draw(self.win)

        self.base.draw(self.win)
        for bird in self.birds:
            bird.draw(self.win)
        
        #Play
        pygame.draw.rect(self.win, self.back_button, [self.x_Play_User, self.y_Play_User ,self.width_Play_User , self.button_height])
        text = STAT_FONT.render(self.message_Play_User,1,self.letter_button)
        self.win.blit(text,(self.x_Play_User+self.xPadding,self.y_Play_User+self.yPadding))

        #Train Genetic Algorithm
        pygame.draw.rect(self.win, self.back_button, [self.x_Train_Genetic, self.y_Train_Genetic ,self.width_Train_Genetic , self.button_height])
        text = STAT_FONT.render(self.message_Train_Genetic,1,self.letter_button)
        self.win.blit(text,(self.x_Train_Genetic+self.xPadding,self.y_Train_Genetic+self.yPadding))
        
        #Play Genetic Algorithm
        pygame.draw.rect(self.win, self.back_button, [self.x_Play_Genetic, self.y_Play_Genetic ,self.width_Play_Genetic , self.button_height])
        text = STAT_FONT.render(self.message_Play_Genetic,1,self.letter_button)
        self.win.blit(text,(self.x_Play_Genetic+self.xPadding,self.y_Play_Genetic+self.yPadding))
        #Play Neural Network
        pygame.draw.rect(self.win, self.back_button, [self.x_Play_Neural, self.y_Play_Neural ,self.width_Play_Neural , self.button_height])
        text = STAT_FONT.render(self.message_Play_Neural,1,self.letter_button)
        self.win.blit(text,(self.x_Play_Neural+self.xPadding,self.y_Play_Neural+self.yPadding))
        pygame.display.update()

        #Play Logistic Regression
        pygame.draw.rect(self.win, self.back_button, [self.x_Play_Logistic, self.y_Play_Logistic ,self.width_Play_Logistic , self.button_height])
        text = STAT_FONT.render(self.message_Play_Logistic,1,self.letter_button)
        self.win.blit(text,(self.x_Play_Logistic+self.xPadding,self.y_Play_Logistic+self.yPadding))

        #Settings
        pygame.draw.rect(self.win, self.back_button, [self.x_Settings, self.y_Settings ,self.width_Settings , self.button_height])
        text = STAT_FONT.render(self.message_Settings,1,self.letter_button)
        self.win.blit(text,(self.x_Settings+self.xPadding,self.y_Settings+self.yPadding))

        #Documentation
        pygame.draw.rect(self.win, self.back_button, [self.x_Documentation, self.y_Documentation ,self.width_Documentation , self.button_height])
        text = STAT_FONT.render(self.message_Documentation,1,self.letter_button)
        self.win.blit(text,(self.x_Documentation+self.xPadding,self.y_Documentation+self.yPadding))


        pygame.display.update()

    def back_ground_play (self):
        #clock delay
        self.clock.tick(self.ticks )
        #Check near pipe
        pipe_ind = 0
        if len(self.pipes) > 1 and self.birds[0].x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                pipe_ind = 1    
        #Move birds
        for x,bird in enumerate(self.birds):
            bird.move()
            #get output to move bird
            output = self.nets[x].activate((bird.y,abs(bird.y-self.pipes[pipe_ind].height),abs(bird.y-self.pipes[pipe_ind].botton)))
            if output[0] > 0.5:
                bird.jump()
        #Var for detection game changes
        add_pipe = False
        rem = []
        #Check collition on pipes
        for pipe in self.pipes:
            #cheack collition for each bird
            for x,bird in enumerate(self.birds):
                # If a bird pass a pipe enable add pipe and passed
                if not pipe.passed and pipe.x <bird.x :
                    pipe.passed = True
                    add_pipe = True
            # If a pipe left scream append a pipe to rem so it can be delate leater
            if pipe.x + pipe.PIPE_TOP.get_width()< 0:
                rem.append(pipe)
            #Move pipes
            pipe.move()
        #Increase score when a pipe is passed
        if add_pipe:
            self.pipes.append(flappyClasses.Pipe(self.PIPES_GAP))
        #Remove pipes
        for r in rem:
            self.pipes.remove (r)
        self.base.move()
    
    def replay_genome_backGround_set(self, genome_path="back.pkl"):
        # Unpickle saved winner
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        # Convert loaded genome into required data structure
        self.genomes = [(1, genome)]
        #Config basic back ground menu 
        for _,g in self.genomes:
            net = neat.nn.FeedForwardNetwork.create(g,self.config)
            self.nets.append(net)
            self.birds.append(flappyClasses.Bird(230,350))
            g.fitness  = 0
            self.ge.append(g)        
        self.pipes = [flappyClasses.Pipe(self.PIPES_GAP)]
        
    def menu(self):
        self.replay_genome_backGround_set()
        while True:
            self.back_ground_play()
            self.draw_window_menu( )
            for event in pygame.event.get():
                self.mouse = pygame.mouse.get_pos()
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                #Play User
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Play_User <= self.mouse[0] <= self.x_Play_User+self.width_Play_User) and (self.y_Play_User <= self.mouse[1] <= self.y_Play_User+self.button_height):
                        print("Play User")
                #Train Genetic Algorithm
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Train_Genetic <= self.mouse[0] <= self.x_Train_Genetic+self.width_Train_Genetic) and (self.y_Train_Genetic <= self.mouse[1] <= self.y_Train_Genetic+self.button_height):
                        print("Play Genetic Algorithm")
                        self.run()
                #Play Genetic Algorithm
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Play_Genetic <= self.mouse[0] <= self.x_Play_Genetic+self.width_Play_Genetic) and (self.y_Play_Genetic <= self.mouse[1] <= self.y_Play_Genetic+self.button_height):
                        print("Play Genetic Algorithm")
                        self.replay_genome()

                #Play Neural Network
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Play_Neural <= self.mouse[0] <= self.x_Play_Neural+self.width_Play_Neural) and (self.y_Play_Neural <= self.mouse[1] <= self.y_Play_Neural+self.button_height):
                        print("Play Neural Network")
                #Play Logistic Regression
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Play_Logistic <= self.mouse[0] <= self.x_Play_Logistic+self.width_Play_Logistic) and (self.y_Play_Logistic <= self.mouse[1] <= self.y_Play_Logistic+self.button_height):
                        print("Play Logistic Regression")
                #Regression
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Settings <= self.mouse[0] <= self.x_Settings+self.width_Settings) and (self.y_Settings <= self.mouse[1] <= self.y_Settings+self.button_height):
                        print("Settings")
                #Documentation
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.x_Documentation <= self.mouse[0] <= self.x_Documentation+self.width_Documentation) and (self.y_Documentation <= self.mouse[1] <= self.y_Documentation+self.button_height):
                        print("Documentation")
if __name__ == '__main__':
    local_dir   = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    game = flappyGame(config_path)
    game.menu()
