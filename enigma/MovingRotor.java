package enigma;

import static enigma.EnigmaException.*;

/** Class that represents a rotating rotor in the enigma machine.
 *  @author danyal
 */
class MovingRotor extends Rotor {

    /** A rotor named NAME whose permutation in its default setting is
     *  PERM, and whose notches are at the positions indicated in NOTCHES.
     *  The Rotor is initally in its 0 setting (first character of its
     *  alphabet).
     */
    MovingRotor(String name, Permutation perm, String notches) {
        super(name, perm);
        _notches = notches;
    }


    @Override
    void advance() {
        int next = setting() + 1;
        set(next);
    }

    /** Checks if rotor rotates.
     * @return boolean
     * */
    boolean rotates() {
        return true;
    }

    /** Checks if rotor is at a notch.
     * @return boolean
     * */
    boolean atNotch() {
        char x = alphabet().toChar(permutation().wrap(setting()));
        if (_notches.equals("")) {
            return false;
        }
        return _notches.contains("" + x);
    }

    /** Converts rotor forwards.
     * @return permutation
     * @param p */
    int convertForward(int p) {
        int conv = permutation().wrap(p + setting());
        int perm = permutation().permute(conv);
        return permutation().wrap(perm - setting());
    }

    /** Converts rotor backwards.
     * @return permutation
     * @param e */
    int convertBackward(int e) {
        int conv = permutation().wrap(e + setting());
        int perm = permutation().invert(conv);
        return permutation().wrap(perm - setting());
    }

    /** The notches. */
    private String _notches;

}
