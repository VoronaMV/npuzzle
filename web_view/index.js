/// <reference path="../lib/typings/jquery/jquery.d.ts"/>

window.onload = () => {

    console.log('afdsf');
    var tiles = [2, 3, 1, 4, 5, 6, 7, 8, 0];
    var text = fs.readFileSync('res.json','utf8');
    console.log(text);

    var $target = undefined;

    var renderTiles = function ($newTarget) {
        $target = $newTarget || $target;

        var $ul = $("<ul>", {
            "class": "n-puzzle"
        });

        $(tiles).each(function (index) {
            var correct = index + 1 == this;
            var cssClass = this == 0 ? "empty" : (correct ? "correct" : "incorrect");

            var $li = $("<li>", {
                "class": cssClass,
                "data-tile": this,
            });
            $li.text(this);
            $li.click({index: index}, shiftTile);
            $ul.append($li);
        })

        var solvable = checkSolvable();

        $target.html($ul);
    };

    var checkSolvable = function () {
        var sum = 0;
        for (var i = 0; i < tiles.length; i++) {

        }
    };

    var shiftTile = function (event) {
        var index = event.data.index;

        var targetIndex = -1;
        if (index - 1 >= 0 && tiles[index - 1] == 0) { // check left
            targetIndex = index - 1;
        } else if (index + 1 < tiles.length && tiles[index + 1] == 0) { // check right
            targetIndex = index + 1;
        } else if (index - 3 >= 0 && tiles[index - 3] == 0) { //check up
            targetIndex = index - 3;
        } else if (index + 3 < tiles.length && tiles[index + 3] == 0) { // check down
            targetIndex = index + 3;
        }

        if (targetIndex != -1) {
            var temp = tiles[targetIndex];
            tiles[targetIndex] = tiles[index];
            tiles[index] = temp;
            renderTiles();
        }

        event.preventDefault();
    };

   renderTiles($('.eight-puzzle'));
};
