# TODO list

This file is a todo list of things with various levels of importance. List is in no particular order.




# Expansion Stuff

## Intra/InterConstruct Transport Cost Tax

Impose specific costs to all unstabilized inputs and outputs for a ComplexConstruct. This would effectively be like declaring "Everything in this construct is moved around via ___", and allow different levels of constructs to have different types of transport costs.

## Surface Restrictions

Classify Constructs by surface, prohibiting specific constructs (some resources, throwing worthless items into the void) from being used depending on the factory surface. Should allow for inter-surface factories.

## Expansion Beacon setups

What updates have to be done to handle diminishing beacons?


# Speedups

## Different solvers?

## Calculation orders

Should be make an array and fill it rather than concat?
np.einsum vs np.kron
Transposing stuff outside function
Combining beacon operations together?


# Misc improvements

## Check and report missing non-optional tag in lua-api: https://lua-api.factorio.com/latest/prototypes/CharacterPrototype.html#mining_speed

## Decorator improvements for: Overloading .add(), type checking interface inputs

## Special case where there are disjoint research bonuses for productivity???

## Fix many typing comments that changed

## Beacon Setups naming in construct

## Changes in cost functions due to effective area and the like

## Lots of naming stuff

Vectors vs columns
setups vs designs
value vs evaluation
etc.
Additionally private vs public class variables

## We aren't using base productivity


# Progression accuracy

## Better catalyst managment

Find a way of handling containers and catalysts properly. Also improve catalyst classification if possible (should iron really be a catalyst?... if not, why not?)
Idea: Catalysts are only catalysts when doing their catalytic process, for example: iron is only catalytic when being used in nuclear fuel. Best method to do this would be to split the catalyst cost off individual constructs, instead detect all catalytic loops and price each for the minimal amount of that each that exists. Catalysts only matter for priced indicies (which effect factory chain addition) and pricing in standard model (not space based models).
Only issue with this is deciding what construct NEEDS the catalytic element made to be allowed (otherwise unpriced costs could end up in our model) (idea: give each item an number, all energy sources have value of 0, otherwise an item has a value equal to 1+ the minimum value of the maximums of making it. The process with the maximum value is the 'final' process)
Doesn't work, breaks linearity


# Requires expansion api

## Expansion Decay constructs

Idea for decay: constructs with no cost that decays a decaying entity into its output.

## Fourth effect: Quality

Implement quality modules.