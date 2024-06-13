# TODO list

This file is a todo list of things with various levels of importance. List is in no particular order.

## Research Affecting Constructs

We need to add in the effect of research on mining productivity and research in labs. My current idea for this is: every mining productivity level needs it own lookuptable. We make a special CompiledConstruct inheriting class that chooses its lookuptable and effect_transform based on the tech level its given. Mining drills will use the lookuptables, while labs only need to multiply the effect_transform.

## Autocalculate factory chain

Add functionality to continously add science factories if more science will be unlocked, otherwise material factories, all the way until 1 material factory after the last possible science factory is done.

## Fix RELEVENT_FLUID_TEMPERATURES orderings

Sometimes RELEVENT_FLUID_TEMPERATURES is given in the wrong order in a function definition.

## Manual Crafting

Sometimes manual crafting is needed (luckily not much in base game after an initial factory). Find some way of calculating when its needed and properly pricing it so that its incredibly punishing to minimize use.

## Optimization of table lookups

Implement the optimizations of table lookups to minimize the amount of possible constructs that are actually calculated for.

## Missing containers

Find containers that are missing from transport cost analysis (atleast sulfuic acid)

## Better catalyst managment

Find a way of handling containers and catalysts properly. Also improve catalyst classification if possible (should iron really be a catalyst?... if not, why not?)

## Expansion Beacon setups

What updates have to be done to handle diminishing beacons?

## Expansion Decay constructs

Idea for decay: constructs with no cost that decays a decaying entity into its output.

## Fourth effect: Quality

Implement quality modules.

## Inverted efficiency evaluations

Instead of just the (most of the time negative) efficiency value, it would be better to output a second value: How many more times expensive the factory would approximately be using this.

