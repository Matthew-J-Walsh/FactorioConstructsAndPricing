# TODO list

This file is a todo list of things with various levels of importance. List is in no particular order.

# Asymptotic speedups

## Optimization of table lookups

Implement the optimizations of table lookups to minimize the amount of possible constructs that are actually calculated for.

## Expansion Beacon setups

What updates have to be done to handle diminishing beacons?

## Retargeting uses old optimization

Add in functionality to let the retargeting to start with a primal guess, specifically for used constructs with potentially well optimized module setups.


# Misc improvements

## Manual Crafting

Sometimes manual crafting is needed (luckily not much in base game after an initial factory). Find some way of calculating when its needed and properly pricing it so that its incredibly punishing to minimize use.


# Progression accuracy

## Better catalyst managment

Find a way of handling containers and catalysts properly. Also improve catalyst classification if possible (should iron really be a catalyst?... if not, why not?)
Idea: Catalysts are only catalysts when doing their catalytic process, for example: iron is only catalytic when being used in nuclear fuel. Best method to do this would be to split the catalyst cost off individual constructs, instead detect all catalytic loops and price each for the minimal amount of that each that exists. Catalysts only matter for priced indicies (which effect factory chain addition) and pricing in standard model (not space based models).
Only issue with this is deciding what construct NEEDS the catalytic element made to be allowed (otherwise unpriced costs could end up in our model) (idea: give each item an number, all energy sources have value of 0, otherwise an item has a value equal to 1+ the minimum value of the maximums of making it. The process with the maximum value is the 'final' process)

## Autocalculate factory chain

Add functionality to continously add science factories if more science will be unlocked, otherwise material factories, all the way until 1 material factory after the last possible science factory is done.


# Requires expansion api

## Expansion Decay constructs

Idea for decay: constructs with no cost that decays a decaying entity into its output.

## Fourth effect: Quality

Implement quality modules.