package enigma;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

/** Class that represents a complete enigma machine.
 *  @author danyal
 */
class Machine {

    /**
     * A new Enigma machine with alphabet ALPHA, 1 < NUMROTORS rotor slots,
     * and 0 <= PAWLS < NUMROTORS pawls.  ALLROTORS contains all the
     * available rotors.
     */
    Machine(Alphabet alpha, int numRotors, int pawls,
            Collection<Rotor> allRotors) {
        _alphabet = alpha;
        _numRotors = numRotors;
        _pawls = pawls;
        _allRotors = new HashMap<>();
        if (_numRotors <= 1) {
            throw new EnigmaException("Not enough Rotors");
        }
        if (_pawls >= _numRotors || _pawls < 0) {
            throw new EnigmaException("Number of pawls is incorrect");
        }
        if (allRotors == null) {
            throw new EnigmaException("Not enough available rotors.");
        }
        for (Rotor i : allRotors) {
            if (!_allRotors.containsValue(i)) {
                _allRotors.put(i.name(), i);
            } else {
                throw new EnigmaException("can't have duplicate rotors");
            }
        }
        _rotors = new Rotor[numRotors()];
    }

    /**
     * Return the number of rotor slots I have.
     */
    int numRotors() {
        return _numRotors;
    }

    /**
     * Return the number pawls (and thus rotating rotors) I have.
     */
    int numPawls() {
        return _pawls;
    }

    /**
     * Set my rotor slots to the rotors named ROTORS from my set of
     * available rotors (ROTORS[0] names the reflector).
     * Initially, all rotors are set at their 0 setting.
     */
    void insertRotors(String[] rotors) {
        int x = 0;
        if (!_allRotors.get(rotors[0]).reflecting()) {
            throw new EnigmaException("First rotor is not a reflector.");
        }
        if (numRotors() != rotors.length) {
            throw new EnigmaException("Not enough rotors.");
        }
        for (String i : rotors) {
            if (_allRotors.containsKey(i)) {
                _rotors[x] = _allRotors.get(i);
                _rotors[x].set(0);
                x++;
            } else {
                throw new EnigmaException("does not contain rotor name.");
            }
        }
    }

    /**
     * Set my rotors according to SETTING, which must be a string of
     * numRotors()-1 characters in my alphabet. The first letter refers
     * to the leftmost rotor setting (not counting the reflector).
     */
    void setRotors(String setting) {
        if (setting.length() != numRotors() - 1) {
            throw new EnigmaException("Numrotors is wrong length.");
        }
        int x = 0;
        char[] set = setting.toCharArray();
        for (char i : set) {
            _rotors[x].set(i);
            x++;
        }
    }

    /**
     * Set the plugboard to PLUGBOARD.
     */
    void setPlugboard(Permutation plugboard) {
        _plugboard = plugboard;
    }

    /** Returns the result of converting the input character C (as an
     *  index in the range 0..alphabet size - 1), after first advancing
     *  the machine. */
    int convert(int c) {
        ArrayList<Rotor> doesMove = new ArrayList<>();
        for (int i = 1; i <= numRotors() - 1; i++) {
            if (!doesMove.contains(i)) {
                if (i != numRotors() - 1) {
                    if (_rotors[i].rotates() && _rotors[i + 1].atNotch()) {
                        doesMove.add(_rotors[i]);
                    }
                } else {
                    doesMove.add(_rotors[i]);
                }
            }
        }
        for (int x = 1; x <= numRotors() - 2; x++) {
            if (doesMove.contains(_rotors[x - 1])
                    && _rotors[x].rotates() && !doesMove.contains(x)) {
                doesMove.add(_rotors[x]);
            }
        }
        for (Rotor x : doesMove) {
            x.advance();
        }

        int newPerm = _plugboard.permute(c);
        for (int i = numRotors() - 1; i >= 0; i--) {
            newPerm = _rotors[i].convertForward(newPerm);
        }
        for (int i = 1; i < numRotors(); i++) {
            newPerm = _rotors[i].convertBackward(newPerm);
        }
        return _plugboard.invert(newPerm);

    }
    /** Returns the encoding/decoding of MSG, updating the state of
     *  the rotors accordingly. */
    String convert(String msg) {
        String total = "";
        char[] message = msg.replaceAll(" ", "").toCharArray();
        for (char i : message) {
            int mesInt = _alphabet.toInt(i);
            total = total + _alphabet.toChar(convert(mesInt));
        }
        return total;
    }

    /** Common alphabet of my rotors. */
    private final Alphabet _alphabet;

    /** Number of Rotors. */
    private int _numRotors;

    /** Number of pawls. */
    private int _pawls;

    /** All possible rotors. */
    private HashMap<String, Rotor> _allRotors;

    /** The plugboard. */
    private Permutation _plugboard;

    /** Used rotors. */
    private Rotor[] _rotors;
}
