#!/usr/bin/env python
import inspect
import typing

import click
import yaml

import cedalion
import cedalion.io
import cedalion.sigproc.tasks
import cedalion.tasks


def is_quantity(hint):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    return (origin is typing.Annotated) and (args[0] is cedalion.Quantity)


def is_dict_of_quantities(origin, args):
    return (origin is dict) and is_quantity(args[1])


def is_list_of_quantities(origin, args):
    return (origin is list) and (len(args) == 1) and is_quantity(args[0])


@click.command()
@click.argument("config", type=click.File("r"), required=True)
@click.argument("src", type=click.Path(exists=True), required=True)
@click.argument("dst", required=True)
def main(config, src, dst):
    config = yaml.safe_load(config)


    rec = cedalion.io.read_snirf(src)[0]

    for task in config["tasks"]:
        if isinstance(task, str):
            print(f"task '{task}' without parameters")

            if (task := cedalion.tasks.task_registry.get(task, None)) is None:
                raise ValueError(f"unknown task {task}")

            task(rec)
        elif isinstance(task, dict):
            assert len(task) == 1
            task_name = next(iter(task.keys()))
            params = next(iter(task.values()))
            print(f"task '{task_name}' with parameters '{params}'")

            if (task := cedalion.tasks.task_registry.get(task_name, None)) is None:
                raise ValueError(f"unknown task {task_name}")

            task_signature = inspect.signature(task)
            task_params = task_signature.parameters.keys()

            param_type_hints = typing.get_type_hints(task, include_extras=True)

            parsed_params = {}

            for param in params:
                assert len(param) == 1
                param_name = next(iter(param.keys()))
                param_value = next(iter(param.values()))

                if param_name not in task_params:
                    raise ValueError(f"unknown param '{param}' for task {task_name}.")

                if param_name not in param_type_hints:
                    parsed_params[param_name] = param_value
                    continue

                param_hint = param_type_hints[param_name]
                hint_origin = typing.get_origin(param_hint)
                hint_args = typing.get_args(param_hint)

                if is_quantity(param_hint):
                    # e.g. typing.Annotated[pint.Quantity, '[length]']
                    dimension = hint_args[1]
                    q = cedalion.Quantity(param_value)
                    q.check(dimension)

                elif is_list_of_quantities(hint_origin, hint_args):
                    # e.g. list[typing.Annotated[pint.Quantity, '[concentration]']]
                    dimension = typing.get_args(hint_args[0])[1]
                    q = [cedalion.Quantity(v) for v in param_value]
                    for v in q:
                        v.check(dimension)

                elif is_dict_of_quantities(hint_origin, hint_args):
                    # e.g. dict[float, typing.Annotated[pint.Quantity, '[time]']]
                    dimension = typing.get_args(hint_args[1])[1]
                    q = {k: cedalion.Quantity(v) for k, v in param_value.items()}
                    for v in q.values():
                        v.check(dimension)
                else:
                    q = param_value

                parsed_params[param_name] = q

                # if isinstance(v, typing._AnnotatedAlias):

            task(rec, **parsed_params)
        else:
            raise ValueError("unexpected task spec.")

    cedalion.io.write_snirf(dst, rec)


if __name__ == "__main__":
    main()
